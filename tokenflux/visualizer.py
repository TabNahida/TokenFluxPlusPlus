"""Vocabulary Distribution Visualization Tool"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import seaborn as sns


class TokenVisualizer:
    """Token and Vocabulary Distribution Visualization Tool"""
    
    def __init__(self, style: str = 'default', figsize: tuple = (12, 8)):
        """
        Initialize the visualizer
        
        Args:
            style: Visualization style ('default', 'dark', 'whitegrid')
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        
        # Set style
        if style == 'dark':
            sns.set_style('darkgrid')
        elif style == 'whitegrid':
            sns.set_style('whitegrid')
            
    def plot_vocab_distribution(self, token_stats: Dict, top_n: int = 50, 
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot vocabulary distribution (Top N frequent tokens)
        
        Args:
            token_stats: Statistics from TokenAnalyzer
            top_n: Show top N frequent tokens
            save_path: Save path (optional)
            
        Returns:
            matplotlib Figure object
        """
        top_tokens = token_stats.get('top_20_tokens', [])
        if len(top_tokens) < top_n:
            top_n = len(top_tokens)
            
        tokens = [t[0][:20] for t in top_tokens[:top_n]]  # Truncate long tokens
        frequencies = [t[1] for t in top_tokens[:top_n]]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create bar chart
        colors = plt.cm.viridis(np.linspace(0, 0.8, top_n))
        bars = ax.bar(range(top_n), frequencies, color=colors)
        
        # Add value labels
        for i, (bar, freq) in enumerate(zip(bars, frequencies)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(frequencies)*0.01,
                   str(freq), ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Token Rank', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Top {top_n} Most Frequent Tokens', fontsize=14, fontweight='bold')
        ax.set_xticks(range(top_n))
        ax.set_xticklabels([f'{i+1}: {t}' for i, t in enumerate(tokens)], rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Vocabulary distribution saved to: {save_path}")
            
        return fig
    
    def plot_token_length_distribution(self, token_stats: Dict, 
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot token length distribution
        
        Args:
            token_stats: Statistics from TokenAnalyzer
            save_path: Save path (optional)
            
        Returns:
            matplotlib Figure object
        """
        length_dist = token_stats.get('token_length_distribution', {})
        
        if not length_dist:
            print("Warning: No token length distribution data available")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=16)
            return fig
            
        lengths = [int(k) for k in length_dist.keys()]
        proportions = [v for v in length_dist.values()]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 创建条形图
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(lengths)))
        bars = ax.bar(lengths, proportions, color=colors, edgecolor='black')
        
        # Add value labels
        for bar, prop in zip(bars, proportions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prop:.2%}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Token Length (characters)', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_title('Token Length Distribution', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(proportions) * 1.2)
        
        # Add average line
        avg_length = token_stats.get('avg_token_length', 0)
        ax.axvline(avg_length, color='red', linestyle='--', 
                  label=f'Average: {avg_length:.2f} chars')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Token length distribution saved to: {save_path}")
            
        return fig
    
    def plot_zipf_law(self, token_stats: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Zipf's law distribution
        
        Args:
            token_stats: Statistics from TokenAnalyzer
            save_path: Save path (optional)
            
        Returns:
            matplotlib Figure object
        """
        zipf_analysis = token_stats.get('zipf_analysis', {})
        
        if not zipf_analysis:
            print("Warning: No Zipf analysis data available")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=16)
            return fig
            
        ranks = zipf_analysis.get('ranks', [])
        frequencies = zipf_analysis.get('frequencies', [])
        zipf_exponent = zipf_analysis.get('zipf_exponent', 1.0)
        r_squared = zipf_analysis.get('r_squared', 0.0)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot actual data (log-log plot)
        ax.loglog(ranks, frequencies, 'o', alpha=0.5, label='Actual Data', markersize=3)
        
        # Plot fitted line
        rank_range = [min(ranks), max(ranks)]
        freq_range = [frequencies[0] * (x / ranks[0]) ** (-zipf_exponent) 
                     for x in rank_range]
        ax.loglog(rank_range, freq_range, 'r-', linewidth=2, 
                 label=f'Fitted (α={zipf_exponent:.2f})')
        
        ax.set_xlabel('Rank (log scale)', fontsize=12)
        ax.set_ylabel('Frequency (log scale)', fontsize=12)
        ax.set_title(f'Zipf\'s Law Distribution\n(R² = {r_squared:.4f})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Zipf's law distribution saved to: {save_path}")
            
        return fig
    
    def plot_vocab_coverage(self, token_stats: Dict, 
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot vocabulary coverage pie chart
        
        Args:
            token_stats: Statistics from TokenAnalyzer
            save_path: Save path (optional)
            
        Returns:
            matplotlib Figure object
        """
        vocab_size = token_stats.get('vocab_size', 0)
        unique_tokens = token_stats.get('unique_tokens', 0)
        vocab_coverage = token_stats.get('vocab_coverage', 0)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建饼图数据
        sizes = [unique_tokens, vocab_size - unique_tokens]
        labels = [f'Used ({unique_tokens})', f'Unused ({vocab_size - unique_tokens})']
        colors = ['#3498db', '#bdc3c7']
        explode = (0.05, 0)
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                         colors=colors, autopct='%1.1f%%',
                                         startangle=90, pctdistance=0.75)
        
        # Add coverage text
        ax.text(0, -1.2, f'Vocabulary Coverage: {vocab_coverage:.2%}',
               ha='center', va='center', fontsize=14, fontweight='bold',
               transform=ax.transAxes)
        
        ax.set_title('Vocabulary Coverage', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Vocabulary coverage saved to: {save_path}")
            
        return fig
    
    def plot_comprehensive_analysis(self, token_stats: Dict, 
                                   save_path: Optional[str] = None) -> List[plt.Figure]:
        """
        Plot comprehensive analysis (all visualizations)
        
        Args:
            token_stats: Statistics from TokenAnalyzer
            save_path: Save path prefix (optional)
            
        Returns:
            List of Figure objects
        """
        figures = []
        
        # Create 2x2 subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Top N vocabulary distribution
        ax1 = fig.add_subplot(2, 2, 1)
        top_tokens = token_stats.get('top_20_tokens', [])[:20]
        tokens = [t[0][:15] for t in top_tokens]
        frequencies = [t[1] for t in top_tokens]
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_tokens)))
        ax1.bar(range(len(top_tokens)), frequencies, color=colors)
        ax1.set_xlabel('Rank', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('Top 20 Most Frequent Tokens', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(top_tokens)))
        ax1.set_xticklabels([f'{i+1}\n{t}' for i, t in enumerate(tokens)], 
                           rotation=0, ha='center', fontsize=8)
        
        # 2. Token length distribution
        ax2 = fig.add_subplot(2, 2, 2)
        length_dist = token_stats.get('token_length_distribution', {})
        if length_dist:
            lengths = [int(k) for k in length_dist.keys()]
            proportions = [v for v in length_dist.values()]
            colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(lengths)))
            ax2.bar(lengths, proportions, color=colors, edgecolor='black')
            ax2.set_xlabel('Token Length', fontsize=10)
            ax2.set_ylabel('Proportion', fontsize=10)
            ax2.set_title('Token Length Distribution', fontsize=12, fontweight='bold')
        
        # 3. Zipf's law
        ax3 = fig.add_subplot(2, 2, 3)
        zipf_analysis = token_stats.get('zipf_analysis', {})
        if zipf_analysis:
            ranks = zipf_analysis.get('ranks', [])[:100]
            frequencies = zipf_analysis.get('frequencies', [])[:100]
            zipf_exponent = zipf_analysis.get('zipf_exponent', 1.0)
            ax3.loglog(ranks, frequencies, 'o', alpha=0.5, label='Actual', markersize=2)
            rank_range = [min(ranks), max(ranks)]
            freq_range = [frequencies[0] * (x / ranks[0]) ** (-zipf_exponent) 
                         for x in rank_range]
            ax3.loglog(rank_range, freq_range, 'r-', linewidth=2, 
                      label=f'Fitted (α={zipf_exponent:.2f})')
            ax3.set_xlabel('Rank (log)', fontsize=10)
            ax3.set_ylabel('Frequency (log)', fontsize=10)
            ax3.set_title("Zipf's Law", fontsize=12, fontweight='bold')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
        
        # 4. Vocabulary coverage
        ax4 = fig.add_subplot(2, 2, 4)
        vocab_size = token_stats.get('vocab_size', 0)
        unique_tokens = token_stats.get('unique_tokens', 0)
        vocab_coverage = token_stats.get('vocab_coverage', 0)
        sizes = [unique_tokens, vocab_size - unique_tokens]
        labels = [f'Used ({unique_tokens})', f'Unused ({vocab_size - unique_tokens})']
        colors = ['#3498db', '#bdc3c7']
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        ax4.set_title(f'Vocabulary Coverage\n({vocab_coverage:.2%})', 
                     fontsize=12, fontweight='bold')
        
        # Add overall title
        fig.suptitle('Comprehensive Token Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        figures.append(fig)
        
        if save_path:
            plt.savefig(f"{save_path}_comprehensive.png", dpi=300, bbox_inches='tight')
            print(f"Comprehensive analysis saved to: {save_path}_comprehensive.png")
            
        return figures