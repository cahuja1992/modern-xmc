"""
LLM-as-Judge Runner

Runs LLM evaluation to validate semantic correctness of (asset, label) pairs.
Supports multiple LLM providers and batching.
"""

from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import json
import time


class SemanticAgreement(Enum):
    """LLM judgment on semantic correctness."""
    YES = "YES"
    NO = "NO"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class LLMJudgment:
    """Result from LLM evaluation."""
    asset_id: str
    label_id: str
    agreement: SemanticAgreement
    confidence: float
    reasoning: Optional[str] = None


class LLMRunner:
    """
    LLM-as-judge runner for offline calibration.
    
    Evaluates semantic correctness of (asset, label) pairs
    without seeing embeddings or confidence scores.
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        provider: str = "openai",
        model: str = "gpt-4",
        batch_size: int = 10,
        max_retries: int = 3,
    ):
        """
        Initialize LLM runner.
        
        Args:
            llm_client: LLM client instance (OpenAI, Anthropic, etc.)
            provider: LLM provider name
            model: Model identifier
            batch_size: Batch size for API calls
            max_retries: Maximum retry attempts
        """
        self.llm_client = llm_client
        self.provider = provider
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
    
    def create_prompt(
        self,
        asset_description: str,
        label_id: str,
        label_definition: Optional[str] = None,
    ) -> str:
        """
        Create evaluation prompt for LLM.
        
        Args:
            asset_description: Text description of the asset
            label_id: Label identifier
            label_definition: Optional definition of the label
        
        Returns:
            Formatted prompt string
        """
        label_info = f"Label: {label_id}"
        if label_definition:
            label_info += f"\nDefinition: {label_definition}"
        
        prompt = f"""You are evaluating whether a label correctly describes an asset.

{label_info}

Asset Description:
{asset_description}

Does this label semantically match this asset?

Respond in JSON format:
{{
  "semantic_agreement": "YES" or "NO" or "UNCERTAIN",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}}

Only use YES or NO. Use UNCERTAIN only if you cannot make a determination.
"""
        return prompt
    
    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured format.
        
        Args:
            response_text: Raw text response from LLM
        
        Returns:
            Parsed dictionary
        """
        # Try to extract JSON from response
        try:
            # Look for JSON block
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
            else:
                # Fallback parsing
                return {"semantic_agreement": "UNCERTAIN", "confidence": 0.5}
        except json.JSONDecodeError:
            return {"semantic_agreement": "UNCERTAIN", "confidence": 0.5}
    
    def evaluate_single(
        self,
        asset_id: str,
        asset_description: str,
        label_id: str,
        label_definition: Optional[str] = None,
    ) -> LLMJudgment:
        """
        Evaluate a single (asset, label) pair.
        
        Args:
            asset_id: Asset identifier
            asset_description: Text description of asset
            label_id: Label identifier
            label_definition: Optional label definition
        
        Returns:
            LLMJudgment object
        """
        prompt = self.create_prompt(asset_description, label_id, label_definition)
        
        # Call LLM (implementation depends on provider)
        if self.llm_client is None:
            # Mock response for testing
            response = {
                "semantic_agreement": "YES",
                "confidence": 0.8,
                "reasoning": "Mock response"
            }
        else:
            response_text = self._call_llm(prompt)
            response = self.parse_response(response_text)
        
        return LLMJudgment(
            asset_id=asset_id,
            label_id=label_id,
            agreement=SemanticAgreement(response.get("semantic_agreement", "UNCERTAIN")),
            confidence=float(response.get("confidence", 0.5)),
            reasoning=response.get("reasoning"),
        )
    
    def evaluate_batch(
        self,
        samples: List[Dict[str, str]],
        asset_descriptions: Dict[str, str],
        label_definitions: Optional[Dict[str, str]] = None,
    ) -> List[LLMJudgment]:
        """
        Evaluate a batch of (asset, label) pairs.
        
        Args:
            samples: List of dicts with 'asset_id' and 'label_id'
            asset_descriptions: Map from asset_id to description
            label_definitions: Optional map from label_id to definition
        
        Returns:
            List of LLMJudgment objects
        """
        label_definitions = label_definitions or {}
        judgments = []
        
        for sample in samples:
            asset_id = sample["asset_id"]
            label_id = sample["label_id"]
            
            asset_desc = asset_descriptions.get(asset_id, "")
            label_def = label_definitions.get(label_id)
            
            judgment = self.evaluate_single(
                asset_id=asset_id,
                asset_description=asset_desc,
                label_id=label_id,
                label_definition=label_def,
            )
            
            judgments.append(judgment)
            
            # Rate limiting
            time.sleep(0.1)
        
        return judgments
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API with retry logic.
        
        Args:
            prompt: Prompt text
        
        Returns:
            Response text
        """
        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    response = self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                    )
                    return response.choices[0].message.content
                
                elif self.provider == "anthropic":
                    response = self.llm_client.messages.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=1024,
                    )
                    return response.content[0].text
                
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return ""
    
    def to_binary_labels(self, judgments: List[LLMJudgment]) -> List[int]:
        """
        Convert LLM judgments to binary labels for calibration.
        
        YES -> 1, NO -> 0, UNCERTAIN is excluded
        
        Args:
            judgments: List of LLM judgments
        
        Returns:
            List of binary labels
        """
        labels = []
        for judgment in judgments:
            if judgment.agreement == SemanticAgreement.YES:
                labels.append(1)
            elif judgment.agreement == SemanticAgreement.NO:
                labels.append(0)
            # Skip UNCERTAIN
        
        return labels
    
    def __repr__(self) -> str:
        return f"LLMRunner(provider={self.provider}, model={self.model})"
