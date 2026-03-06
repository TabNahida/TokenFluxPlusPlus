"""Token Statistics Analysis Tool"""

from typing import Dict, List, Tuple, Optional
import json
from collections import Counter
import numpy as np


class TokenAnalyzer:
    """Token Statistics Analysis Tool
    
    Provides the following functionality:
    - Token frequency statistics
    - Vocabulary coverage analysis
    - OOV (Out-of-Vocabulary) rate calculation
    - Token length distribution
    - Vocabulary distribution visualization
    """
    
    def __init__(self, tokenizer):
        """
        Initialize the analyzer
        
        Args:
            tokenizer: TokenFlux Tokenizer instance or tokenizer.json path
        """
        # If string path is provided, create Tokenizer instance
        if isinstance(tokenizer, str):
            import tokenflux
            self.tokenizer = tokenflux.Tokenizer(tokenizer)
        else:
            self.tokenizer = tokenizer
        self._vocab_size = None
        self._id_to_token_map = None
        
    def _build_id_to_token_map(self) -> Dict[int, str]:
        """Build id to token mapping table"""
        if self._id_to_token_map is not None:
            return self._id_to_token_map
            
        # Get vocabulary from tokenizer.json
        import os
        tokenizer_json_path = self.tokenizer.tokenizer_path
        
        if os.path.exists(tokenizer_json_path):
            with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
                
            # Get vocab (format is {token: id}, need to reverse to {id: token})
            vocab = tokenizer_data.get('model', {}).get('vocab', {})
            self._vocab_size = len(vocab)
            self._id_to_token_map = {v: k for k, v in vocab.items()}
        else:
            raise FileNotFoundError(f"Tokenizer JSON file not found: {tokenizer_json_path}")
            
        return self._id_to_token_map
    
    def analyze_tokens(self, texts: List[str]) -> Dict:
        """
        Analyze token statistics for texts
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary containing statistics
        """
        # Encode texts
        encoded_batch = self.tokenizer.encode_batch(texts)
        
        # Flatten all token ids
        all_tokens = []
        for token_ids in encoded_batch:
            all_tokens.extend(token_ids)
            
        # Count token frequency
        token_counter = Counter(all_tokens)
        
        # Build id to token mapping
        id_to_token = self._build_id_to_token_map()
        
        # Convert to token string frequency
        token_str_counter = Counter()
        for token_id, count in token_counter.items():
            token_str = id_to_token.get(token_id, f"<ID_{token_id}>")
            token_str_counter[token_str] = count
            
        # Calculate statistics
        total_tokens = len(all_tokens)
        unique_tokens = len(token_counter)
        vocab_size = self._vocab_size
        
        # Vocabulary coverage
        vocab_coverage = unique_tokens / vocab_size if vocab_size > 0 else 0
        
        # Calculate token length distribution
        token_lengths = []
        for token_id in token_counter.keys():
            token_str = id_to_token.get(token_id, "")
            token_lengths.append(len(token_str))
            
        # Calculate Top N frequent tokens
        top_n_tokens = token_str_counter.most_common(20)
        
        # Calculate Zipf's law fit
        zipf_analysis = self._analyze_zipf_distribution(token_counter, id_to_token)
        
        return {
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'vocab_size': vocab_size,
            'vocab_coverage': vocab_coverage,
            'token_frequency': dict(token_str_counter),
            'top_20_tokens': top_n_tokens,
            'token_length_distribution': self._compute_length_distribution(token_lengths),
            'zipf_analysis': zipf_analysis,
            'avg_token_length': np.mean(token_lengths) if token_lengths else 0,
            'median_token_length': np.median(token_lengths) if token_lengths else 0,
        }
    
    def _compute_length_distribution(self, lengths: List[int]) -> Dict[str, float]:
        """Compute token length distribution"""
        if not lengths:
            return {}
            
        length_counter = Counter(lengths)
        total = sum(length_counter.values())
        
        return {
            str(length): count / total
            for length, count in sorted(length_counter.items())
        }
    
    def _analyze_zipf_distribution(self, token_counter: Counter, id_to_token: Dict[int, str]) -> Dict:
        """Analyze Zipf's law distribution"""
        # Sort by frequency
        sorted_tokens = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate ranks and frequencies
        ranks = []
        frequencies = []
        for rank, (token_id, freq) in enumerate(sorted_tokens, 1):
            ranks.append(rank)
            frequencies.append(freq)
            
        # Linear regression fit for log(rank) vs log(frequency)
        if len(ranks) > 1:
            log_ranks = np.log(ranks)
            log_freqs = np.log(frequencies)
            
            # Calculate slope (Zipf exponent)
            zipf_exponent = -np.polyfit(log_ranks, log_freqs, 1)[0]
            
            # Calculate R²
            predicted = -zipf_exponent * log_ranks + np.polyfit(log_ranks, log_freqs, 1)[1]
            ss_res = np.sum((log_freqs - predicted) ** 2)
            ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            zipf_exponent = 1.0
            r_squared = 0.0
            
        return {
            'zipf_exponent': float(zipf_exponent),
            'r_squared': float(r_squared),
            'ranks': ranks[:100],  # Top 100
            'frequencies': frequencies[:100],
        }
    
    def compute_oov_rate(self, texts: List[str], reserved_tokens: Optional[set] = None) -> float:
        """
        Calculate OOV (Out-of-Vocabulary) rate
        
        Args:
            texts: List of texts
            reserved_tokens: Reserved token set (e.g., special tokens)
            
        Returns:
            OOV rate (between 0.0 and 1.0)
        """
        if reserved_tokens is None:
            reserved_tokens = set()
            
        # Get all tokens from vocabulary
        id_to_token = self._build_id_to_token_map()
        vocab_tokens = set(id_to_token.values())
        
        # Encode texts
        encoded_batch = self.tokenizer.encode_batch(texts)
        
        # Count all tokens
        all_tokens = []
        for token_ids in encoded_batch:
            all_tokens.extend(token_ids)
            
        # Count OOV tokens
        oov_count = sum(1 for token_id in all_tokens 
                       if token_id not in id_to_token or 
                       id_to_token.get(token_id) not in vocab_tokens - reserved_tokens)
                       
        return oov_count / len(all_tokens) if all_tokens else 0.0
    
    def get_token_statistics(self, texts: List[str]) -> Dict:
        """
        Get detailed token statistics
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary containing detailed statistics
        """
        stats = self.analyze_tokens(texts)
        
        # Add more information
        stats['per_document_stats'] = []
        encoded_batch = self.tokenizer.encode_batch(texts)
        
        for i, (text, token_ids) in enumerate(zip(texts, encoded_batch)):
            doc_stats = {
                'doc_id': i,
                'char_length': len(text),
                'token_count': len(token_ids),
                'chars_per_token': len(text) / len(token_ids) if token_ids else 0,
            }
            stats['per_document_stats'].append(doc_stats)
            
        return stats