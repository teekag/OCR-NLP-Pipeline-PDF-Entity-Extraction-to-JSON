"""
Entity Extractor Module
----------------------
Provides entity extraction capabilities using various NLP models.
Supports spaCy, Flair, and Transformers-based entity recognition.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json

import spacy
from spacy.tokens import Doc
import flair
from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EntityExtractor(ABC):
    """Abstract base class for entity extractors."""
    
    @abstractmethod
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities with metadata
        """
        pass
    
    @abstractmethod
    def get_supported_entity_types(self) -> List[str]:
        """
        Get list of supported entity types.
        
        Returns:
            List of entity type strings
        """
        pass
    
    def extract_entities_from_documents(self, documents: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Extract entities from multiple documents.
        
        Args:
            documents: List of text documents
            
        Returns:
            List of entity lists, one per document
        """
        results = []
        for doc in documents:
            entities = self.extract_entities(doc)
            results.append(entities)
        return results
    
    def save_entities_to_json(self, entities: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
        """
        Save extracted entities to JSON file.
        
        Args:
            entities: List of extracted entities
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entities, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(entities)} entities to {output_path}")


class SpacyEntityExtractor(EntityExtractor):
    """Entity extractor using spaCy models."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize spaCy entity extractor.
        
        Args:
            model_name: Name of spaCy model to use
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"spaCy model {model_name} not found. Using blank model.")
            self.nlp = spacy.blank("en")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities with metadata
        """
        if not text:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity = {
                'text': ent.text,
                'label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char,
                'source': 'spacy'
            }
            entities.append(entity)
        
        return entities
    
    def get_supported_entity_types(self) -> List[str]:
        """
        Get list of supported entity types in spaCy.
        
        Returns:
            List of entity type strings
        """
        return list(self.nlp.pipe_labels['ner'])
    
    def extract_entities_with_context(self, text: str, context_window: int = 10) -> List[Dict[str, Any]]:
        """
        Extract entities with surrounding context.
        
        Args:
            text: Input text
            context_window: Number of characters to include before and after entity
            
        Returns:
            List of entities with context
        """
        entities = self.extract_entities(text)
        
        for entity in entities:
            start = max(0, entity['start_char'] - context_window)
            end = min(len(text), entity['end_char'] + context_window)
            
            entity['context'] = text[start:end]
            entity['context_start'] = start
            entity['context_end'] = end
        
        return entities


class FlairEntityExtractor(EntityExtractor):
    """Entity extractor using Flair models."""
    
    def __init__(self, model_name: str = "ner"):
        """
        Initialize Flair entity extractor.
        
        Args:
            model_name: Name of Flair model to use
        """
        try:
            self.tagger = SequenceTagger.load(model_name)
            logger.info(f"Loaded Flair model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading Flair model: {str(e)}")
            raise
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using Flair.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities with metadata
        """
        if not text:
            return []
        
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        
        entities = []
        for entity in sentence.get_spans('ner'):
            entity_dict = {
                'text': entity.text,
                'label': entity.tag,
                'start_char': entity.start_pos,
                'end_char': entity.end_pos,
                'confidence': entity.score,
                'source': 'flair'
            }
            entities.append(entity_dict)
        
        return entities
    
    def get_supported_entity_types(self) -> List[str]:
        """
        Get list of supported entity types in Flair.
        
        Returns:
            List of entity type strings
        """
        # This is a simplified approach - in reality, you'd need to
        # extract this from the model's tag dictionary
        return ["PER", "LOC", "ORG", "MISC"]


class TransformersEntityExtractor(EntityExtractor):
    """Entity extractor using Hugging Face Transformers models."""
    
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """
        Initialize Transformers entity extractor.
        
        Args:
            model_name: Name of Transformers model to use
        """
        try:
            self.ner_pipeline = pipeline(
                "ner", 
                model=model_name, 
                aggregation_strategy="simple"
            )
            logger.info(f"Loaded Transformers model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading Transformers model: {str(e)}")
            raise
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using Transformers.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities with metadata
        """
        if not text:
            return []
        
        try:
            # The pipeline might fail on very long texts, so we'll split it
            # This is a simple approach - in production, you'd want a more sophisticated chunking strategy
            max_length = 512
            if len(text) > max_length:
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                all_entities = []
                offset = 0
                
                for chunk in chunks:
                    entities = self.ner_pipeline(chunk)
                    
                    # Adjust offsets for the chunk
                    for entity in entities:
                        entity['start'] += offset
                        entity['end'] += offset
                    
                    all_entities.extend(entities)
                    offset += len(chunk)
                
                entities = all_entities
            else:
                entities = self.ner_pipeline(text)
            
            # Convert to standard format
            # NOTE: Different models return different formats, we normalize here
            results = []
            for entity in entities:
                entity_dict = {
                    'text': entity['word'],
                    'label': entity['entity_group'],
                    'start_char': entity['start'],
                    'end_char': entity['end'],
                    'confidence': entity['score'],
                    'source': 'transformers'
                }
                results.append(entity_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting entities with Transformers: {str(e)}")
            return []
    
    def get_supported_entity_types(self) -> List[str]:
        """
        Get list of supported entity types in Transformers model.
        
        Returns:
            List of entity type strings
        """
        # This is a simplified approach - in reality, you'd need to
        # extract this from the model's config or label mapping
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class CustomEntityExtractor(EntityExtractor):
    """Custom entity extractor for domain-specific entities."""
    
    def __init__(self, patterns: Dict[str, List[str]] = None, rules: Dict[str, List[Dict]] = None):
        """
        Initialize custom entity extractor.
        
        Args:
            patterns: Dictionary mapping entity types to regex patterns
            rules: Dictionary mapping entity types to extraction rules
        """
        import re
        
        self.patterns = patterns or {}
        self.rules = rules or {}
        
        # Compile regex patterns
        self.compiled_patterns = {}
        for entity_type, pattern_list in self.patterns.items():
            self.compiled_patterns[entity_type] = [re.compile(pattern, re.IGNORECASE) for pattern in pattern_list]
        
        logger.info(f"Initialized CustomEntityExtractor with {len(self.patterns)} pattern types and {len(self.rules)} rule types")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using custom patterns and rules.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities with metadata
        """
        if not text:
            return []
        
        entities = []
        
        # Extract entities using regex patterns
        for entity_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity = {
                        'text': match.group(),
                        'label': entity_type,
                        'start_char': match.start(),
                        'end_char': match.end(),
                        'source': 'custom_pattern'
                    }
                    entities.append(entity)
        
        # Apply extraction rules (simplified implementation)
        # In a real system, rules would be more sophisticated
        for entity_type, rule_list in self.rules.items():
            for rule in rule_list:
                if 'prefix' in rule and 'suffix' in rule:
                    prefix = rule['prefix']
                    suffix = rule['suffix']
                    
                    # Find all occurrences of prefix...suffix
                    start_pos = 0
                    while True:
                        start = text.find(prefix, start_pos)
                        if start == -1:
                            break
                        
                        end = text.find(suffix, start + len(prefix))
                        if end == -1:
                            break
                        
                        # Extract the entity between prefix and suffix
                        entity_text = text[start + len(prefix):end].strip()
                        if entity_text:
                            entity = {
                                'text': entity_text,
                                'label': entity_type,
                                'start_char': start + len(prefix),
                                'end_char': end,
                                'source': 'custom_rule'
                            }
                            entities.append(entity)
                        
                        start_pos = end + len(suffix)
        
        return entities
    
    def get_supported_entity_types(self) -> List[str]:
        """
        Get list of supported entity types.
        
        Returns:
            List of entity type strings
        """
        entity_types = list(set(list(self.patterns.keys()) + list(self.rules.keys())))
        return entity_types
    
    @classmethod
    def create_from_config(cls, config_path: Union[str, Path]) -> 'CustomEntityExtractor':
        """
        Create a CustomEntityExtractor from a configuration file.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            CustomEntityExtractor instance
        """
        config_path = Path(config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        patterns = config.get('patterns', {})
        rules = config.get('rules', {})
        
        return cls(patterns=patterns, rules=rules)


class EntityExtractorFactory:
    """Factory for creating entity extractors."""
    
    @staticmethod
    def create_extractor(extractor_type: str, **kwargs) -> EntityExtractor:
        """
        Create an entity extractor of the specified type.
        
        Args:
            extractor_type: Type of extractor ('spacy', 'flair', 'transformers', 'custom')
            **kwargs: Additional arguments for the specific extractor
            
        Returns:
            EntityExtractor instance
        """
        if extractor_type.lower() == 'spacy':
            return SpacyEntityExtractor(**kwargs)
        elif extractor_type.lower() == 'flair':
            return FlairEntityExtractor(**kwargs)
        elif extractor_type.lower() == 'transformers':
            return TransformersEntityExtractor(**kwargs)
        elif extractor_type.lower() == 'custom':
            return CustomEntityExtractor(**kwargs)
        else:
            raise ValueError(f"Unsupported entity extractor type: {extractor_type}")


class EntityMerger:
    """Merges entities from multiple extractors."""
    
    def __init__(self, extractors: List[EntityExtractor], priority_order: List[str] = None):
        """
        Initialize entity merger.
        
        Args:
            extractors: List of entity extractors
            priority_order: List of extractor source names in priority order (highest first)
        """
        self.extractors = extractors
        self.priority_order = priority_order or ['custom_rule', 'custom_pattern', 'transformers', 'flair', 'spacy']
    
    def _get_priority(self, source: str) -> int:
        """Get priority for a source."""
        try:
            return self.priority_order.index(source)
        except ValueError:
            return len(self.priority_order)
    
    def _entities_overlap(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
        """Check if two entities overlap."""
        return (
            (entity1['start_char'] <= entity2['start_char'] < entity1['end_char']) or
            (entity1['start_char'] < entity2['end_char'] <= entity1['end_char']) or
            (entity2['start_char'] <= entity1['start_char'] < entity2['end_char']) or
            (entity2['start_char'] < entity1['end_char'] <= entity2['end_char'])
        )
    
    def merge_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract and merge entities from all extractors.
        
        Args:
            text: Input text
            
        Returns:
            List of merged entities
        """
        if not text:
            return []
        
        # Extract entities from all extractors
        all_entities = []
        for extractor in self.extractors:
            entities = extractor.extract_entities(text)
            all_entities.extend(entities)
        
        # Sort by priority and position
        all_entities.sort(key=lambda e: (self._get_priority(e['source']), e['start_char']))
        
        # Resolve overlaps by keeping higher priority entities
        merged_entities = []
        for entity in all_entities:
            # Check if this entity overlaps with any already accepted entity
            overlaps = False
            for accepted_entity in merged_entities:
                if self._entities_overlap(entity, accepted_entity):
                    overlaps = True
                    break
            
            if not overlaps:
                merged_entities.append(entity)
        
        return merged_entities


# Example usage
if __name__ == "__main__":
    # Sample text
    sample_text = """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
    Tim Cook is the CEO of Apple since 2011. The company was founded by Steve Jobs, Steve Wozniak,
    and Ronald Wayne in April 1976. Apple's revenue was $274.5 billion in 2020.
    Contact them at info@apple.com or visit their website at https://www.apple.com.
    """
    
    # Create entity extractors
    spacy_extractor = EntityExtractorFactory.create_extractor('spacy')
    
    # Extract entities
    entities = spacy_extractor.extract_entities(sample_text)
    
    # Print results
    print(f"Found {len(entities)} entities:")
    for entity in entities:
        print(f"- {entity['text']} ({entity['label']})")
    
    # Create custom extractor with patterns
    custom_patterns = {
        'EMAIL': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
        'URL': [r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+']
    }
    
    custom_extractor = EntityExtractorFactory.create_extractor('custom', patterns=custom_patterns)
    custom_entities = custom_extractor.extract_entities(sample_text)
    
    print(f"\nFound {len(custom_entities)} custom entities:")
    for entity in custom_entities:
        print(f"- {entity['text']} ({entity['label']})")
    
    # Merge entities from both extractors
    merger = EntityMerger([spacy_extractor, custom_extractor])
    merged_entities = merger.merge_entities(sample_text)
    
    print(f"\nAfter merging: {len(merged_entities)} entities")
