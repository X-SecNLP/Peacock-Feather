import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sentence_transformers import SentenceTransformer, util

class SemanticNicheExplorer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print("Initializing Semantic Engine...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device)
        self.words = []
        self.embeddings = None

    def load_custom_words(self, word_list):
        """Option A: Pass a direct list of strings."""
        self.words = sorted(list(set([w.lower() for w in word_list])))
        self._encode_space()

    def fit_text_file(self, file_path):
        """Option B: Extract unique words from a text file."""
        unique_words = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    found = re.findall(r'\b\w+\b', line.lower())
                    unique_words.update(found)
            self.words = sorted(list(unique_words))
            self._encode_space()
        except FileNotFoundError:
            print(f"Error: {file_path} not found.")

    def _encode_space(self):
        """Internal helper to calculate embeddings."""
        if not self.words:
            raise ValueError("Word list is empty!")
        print(f"Space loaded: {len(self.words)} unique words. Encoding...")
        self.embeddings = self.model.encode(self.words, convert_to_tensor=True).to(self.device)

    def run_evolution(self, query, generations=40, pop_size=40, sigma_share=0.2, mutation_rate=0.2):
        if self.embeddings is None: return []
        
        query_embedding = self.model.encode(query, convert_to_tensor=True).to(self.device)
        pop_indices = np.random.randint(0, len(self.words), pop_size)
        history = []

        for gen in range(generations):
            current_embs = self.embeddings[pop_indices]
            raw_fitness = util.cos_sim(query_embedding, current_embs)[0].cpu().numpy()
            
            # Niche Sharing Logic
            diff = current_embs.unsqueeze(1) - current_embs.unsqueeze(0)
            dist_matrix = torch.norm(diff, dim=2).cpu().numpy()
            sharing_matrix = np.maximum(0, 1 - (dist_matrix / sigma_share))
            niche_count = np.sum(sharing_matrix, axis=1)
            shared_fitness = raw_fitness / (niche_count + 1e-9)

            # Evolutionary Steps
            probs = (shared_fitness - shared_fitness.min() + 1e-6)
            probs /= probs.sum()
            pop_indices = pop_indices[np.random.choice(pop_size, size=pop_size, p=probs)]

            # Mutation
            mask = np.random.rand(pop_size) < mutation_rate
            if np.any(mask):
                pop_indices[mask] = np.random.randint(0, len(self.words), np.sum(mask))
            
            history.append({
                'gen': gen, 'indices': pop_indices.copy(), 'fitness': raw_fitness,
                'avg_fitness': np.mean(raw_fitness),
                'best_word': self.words[pop_indices[np.argmax(raw_fitness)]]
            })
        return history

    def generate_artifact(self, history, output_gif='result.gif'):
        if not history: return
        fig = plt.figure(figsize=(10, 10), facecolor='#050505')
        ax = plt.subplot(111, projection='polar')
        ax.set_facecolor('#050505')
        plt.axis('off')

        scat = ax.scatter([], [], s=[], c=[], cmap='cool', edgecolors='white', linewidth=0.3, alpha=0.8)
        txt_word = ax.text(0, 0, "", color="white", ha="center", va="center", fontsize=24, fontweight='bold')
        txt_info = ax.text(0, -1.2, "", color="#00FFCC", ha="center", va="center", fontsize=10, family='monospace')
        lines = [ax.plot([], [], color='white', alpha=0.1, lw=0.5)[0] for _ in range(10)]

        def update(frame):
            data = history[frame]
            theta = (data['indices'] / len(self.words)) * 2 * np.pi
            r = 1.0 - (data['fitness'] * 0.9)
            scat.set_offsets(np.c_[theta, r])
            scat.set_sizes(data['fitness'] * 700 + 20)
            scat.set_array(data['fitness'])
            
            top_idx = np.argsort(data['fitness'])[-5:]
            for i, line in enumerate(lines[:len(top_idx)]):
                idx = top_idx[i]
                line.set_data([theta[idx], 0], [r[idx], 0])
            
            txt_word.set_text(data['best_word'].upper())
            txt_info.set_text(f"GEN: {data['gen']:02d} | SIMILARITY: {data['avg_fitness']:.4f}")
            return [scat, txt_word, txt_info] + lines

        ani = FuncAnimation(fig, update, frames=len(history), interval=100, blit=True)
        ani.save(output_gif, writer=PillowWriter(fps=10))
        plt.show()

# --- HOW TO CUSTOMIZE ---
if __name__ == "__main__":
    explorer = SemanticNicheExplorer()

    # CUSTOMIZE HERE: Add your own words to this list!
    my_custom_vocabulary = []
    
    # Use Option A:
    # explorer.load_custom_words(my_custom_vocabulary)
    
    # OR Use Option B (Uncomment to use a real file):
    explorer.fit_text_file('/content/lyrics.txt')

    # Run the simulation
    history = explorer.run_evolution(query="dich", generations=100)
    explorer.generate_artifact(history)
