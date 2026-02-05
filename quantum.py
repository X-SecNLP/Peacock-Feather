import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sentence_transformers import SentenceTransformer, util

class QuantumPhaseExplorer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print("Initializing Quantum Semantic Engine...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device)
        self.words = []
        self.embeddings = None

    def load_custom_words(self, word_list):
        self.words = sorted(list(set([w.lower() for w in word_list])))
        self._encode_space()

    def fit_text_file(self, file_path):
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
        if not self.words: raise ValueError("Word list is empty!")
        print(f"Quantum Space loaded: {len(self.words)} qubits. Encoding...")
        self.embeddings = self.model.encode(self.words, convert_to_tensor=True).to(self.device)

    def run_quantum_evolution(self, query, generations=60, pop_size=50):
        if self.embeddings is None: return []
        
        query_embedding = self.model.encode(query, convert_to_tensor=True).to(self.device)
        pop_indices = np.random.randint(0, len(self.words), pop_size)
        history = []

        for gen in range(generations):
            # 量子临界参数 g: 从 2.0 (无序) 演化到 0.0 (有序)
            g = 2.0 * (1 - gen / generations)
            
            current_embs = self.embeddings[pop_indices]
            fitness = util.cos_sim(query_embedding, current_embs)[0].cpu().numpy()
            
            # 量子相变核心：使用玻尔兹曼权重模拟量子态概率
            # 这里的 g 扮演了类似温度但本质是量子涨落的角色
            beta = 1.0 / (g + 1e-2) 
            exp_fitness = np.exp(fitness * beta)
            probs = exp_fitness / exp_fitness.sum()
            
            # 采样下一代量子分布
            pop_indices = pop_indices[np.random.choice(pop_size, size=pop_size, p=probs)]

            # 量子隧穿效应 (Quantum Tunneling): g 越高，跳出局部最优的概率越大
            tunneling_prob = np.tanh(g) * 0.5
            mask = np.random.rand(pop_size) < tunneling_prob
            if np.any(mask):
                pop_indices[mask] = np.random.randint(0, len(self.words), np.sum(mask))
            
            # 定义相态
            if g > 1.2: phase = "DISORDERED"
            elif g > 0.7: phase = "CRITICAL"
            else: phase = "ORDERED"

            history.append({
                'gen': gen, 'indices': pop_indices.copy(), 'fitness': fitness,
                'g': g, 'phase': phase,
                'best_word': self.words[pop_indices[np.argmax(fitness)]]
            })
        return history

    def generate_quantum_artifact(self, history, output_gif='quantum_phase_transition.gif'):
        if not history: return
        fig = plt.figure(figsize=(10, 10), facecolor='#00050a')
        ax = plt.subplot(111, projection='polar')
        ax.set_facecolor('#00050a')
        plt.axis('off')

        scat = ax.scatter([], [], s=[], c=[], cmap='magma', edgecolors='none', alpha=0.7)
        txt_word = ax.text(0, 0, "", color="cyan", ha="center", va="center", fontsize=26, fontweight='bold', alpha=0.8)
        txt_phase = ax.text(0, 1.3, "", color="white", ha="center", va="center", fontsize=12, family='monospace')
        
        # 模拟量子场线的视觉效果
        field_lines = [ax.plot([], [], color='cyan', alpha=0.2, lw=0.5)[0] for _ in range(8)]

        def update(frame):
            data = history[frame]
            theta = (data['indices'] / len(self.words)) * 2 * np.pi
            r = 1.2 - (data['fitness'] * 1.0) # 越接近中心，适应度越高
            
            # 更新粒子
            scat.set_offsets(np.c_[theta, r])
            scat.set_sizes(data['fitness'] * 800 * (1.5 - data['g']/2))
            scat.set_array(data['fitness'])
            
            # 临界区视觉：增加连线密度模拟量子纠缠感
            top_indices = np.argsort(data['fitness'])[-8:]
            for i, line in enumerate(field_lines):
                idx = top_indices[i]
                # 在有序态时，线条收缩向中心
                line.set_data([theta[idx], 0], [r[idx], 0])
                line.set_alpha(0.5 * (1 - data['g']/2))

            txt_word.set_text(data['best_word'].upper())
            txt_phase.set_text(f"PHASE: {data['phase']} | FIELD STRENGTH g: {data['g']:.3f}")
            
            # 动态调整发光感
            if data['phase'] == "CRITICAL":
                txt_word.set_color("#FF00FF") # 临界区紫色发光
            elif data['phase'] == "ORDERED":
                txt_word.set_color("#00FFFF") # 有序态青色
            
            return [scat, txt_word, txt_phase] + field_lines

        ani = FuncAnimation(fig, update, frames=len(history), interval=100, blit=True)
        ani.save(output_gif, writer=PillowWriter(fps=10))
        plt.show()

# --- 执行 ---
if __name__ == "__main__":
    explorer = QuantumPhaseExplorer()
    
    # 这里可以传入你的词库，或者手动输入
    vocabulary = ["quantum", "bit", "entanglement", "phase", "transition", "superposition", "wave", "particle", "gravity", "space", "time", "schrodinger", "heisenberg", "field", "energy"]
    # 模拟一个较大的随机词库增强视觉效果
    vocabulary += [f"word_{i}" for i in range(500)] 
    
    explorer.load_custom_words(vocabulary)
    
    # query="gravity" 模拟寻找特定量子态的过程
    history = explorer.run_quantum_evolution(query="quantum", generations=80)
    explorer.generate_quantum_artifact(history)
