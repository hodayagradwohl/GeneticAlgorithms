"""
Main file for Magic Square Genetic Algorithm System
Computational Biology - Exercise 2
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time

from standard_magic_square import StandardMagicSquare
from perfect_magic_square import PerfectMagicSquare

class MagicSquareGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Magic Square - Genetic Algorithms")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Variables for tracking execution
        self.is_running = False
        self.current_solver = None
        self.results = {}
        
        self.create_widgets()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_window_close)
        
        
    def on_window_close(self):
        """Called when the window is closed."""
        if self.is_running:
            if messagebox.askokcancel("Warning", "Algorithms are still running. Do you want to stop and close?"):
                self.is_running = False
                self.root.destroy()
        else:
            self.root.destroy()
        
    def create_widgets(self):
        # Main title
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(pady=10)
        
        title_label = tk.Label(title_frame, text="Magic Square - Genetic Algorithms", 
                              font=('Arial', 20, 'bold'), bg='#f0f0f0')
        title_label.pack()
        
        # Settings frame
        settings_frame = tk.LabelFrame(self.root, text="Settings", font=('Arial', 12, 'bold'),
                                     bg='#f0f0f0', padx=10, pady=10)
        settings_frame.pack(pady=10, padx=20, fill='x')
        
        # First row - Square type and size
        row1 = tk.Frame(settings_frame, bg='#f0f0f0')
        row1.pack(fill='x', pady=5)
        
        tk.Label(row1, text="Square Type:", font=('Arial', 10), bg='#f0f0f0').pack(side='left')
        
        self.square_type = tk.StringVar(value="standard")
        tk.Radiobutton(row1, text="Standard Magic Square", variable=self.square_type, 
                      value="standard", bg='#f0f0f0', command=self.on_square_type_change).pack(side='left', padx=10)
        tk.Radiobutton(row1, text="Perfect Magic Square", variable=self.square_type, 
                      value="perfect", bg='#f0f0f0', command=self.on_square_type_change).pack(side='left', padx=10)
        
        tk.Label(row1, text="Size (N):", font=('Arial', 10), bg='#f0f0f0').pack(side='left', padx=(20,5))
        self.size_var = tk.StringVar(value="3")
        size_combo = ttk.Combobox(row1, textvariable=self.size_var, values=["3", "4", "5", "8"], width=5)
        size_combo.pack(side='left')
        
        # Second row - Algorithm parameters
        row2 = tk.Frame(settings_frame, bg='#f0f0f0')
        row2.pack(fill='x', pady=5)
        
        tk.Label(row2, text="Population Size:", font=('Arial', 10), bg='#f0f0f0').pack(side='left')
        self.pop_size_var = tk.StringVar(value="100")
        tk.Entry(row2, textvariable=self.pop_size_var, width=8).pack(side='left', padx=5)
        
        tk.Label(row2, text="Max Generations:", font=('Arial', 10), bg='#f0f0f0').pack(side='left', padx=(20,5))
        self.max_gen_var = tk.StringVar(value="1000")
        tk.Entry(row2, textvariable=self.max_gen_var, width=8).pack(side='left', padx=5)
        
        tk.Label(row2, text="Mutation Rate:", font=('Arial', 10), bg='#f0f0f0').pack(side='left', padx=(20,5))
        self.mutation_rate_var = tk.StringVar(value="0.1")
        tk.Entry(row2, textvariable=self.mutation_rate_var, width=8).pack(side='left', padx=5)
        
        # Algorithm selection frame
        algo_frame = tk.LabelFrame(self.root, text="Algorithm Selection", font=('Arial', 12, 'bold'),
                                  bg='#f0f0f0', padx=10, pady=10)
        algo_frame.pack(pady=10, padx=20, fill='x')
        
        # Checkboxes for algorithms
        self.run_classic = tk.BooleanVar(value=True)
        self.run_darwinian = tk.BooleanVar(value=True)
        self.run_lamarckian = tk.BooleanVar(value=True)
        
        tk.Checkbutton(algo_frame, text="Classic Genetic Algorithm", variable=self.run_classic,
                      bg='#f0f0f0', font=('Arial', 10)).pack(side='left', padx=20)
        tk.Checkbutton(algo_frame, text="Darwinian Algorithm", variable=self.run_darwinian,
                      bg='#f0f0f0', font=('Arial', 10)).pack(side='left', padx=20)
        tk.Checkbutton(algo_frame, text="Lamarckian Algorithm", variable=self.run_lamarckian,
                      bg='#f0f0f0', font=('Arial', 10)).pack(side='left', padx=20)
        
        # Control buttons
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=20)
        
        self.run_button = tk.Button(button_frame, text="Run Algorithms", font=('Arial', 12, 'bold'),
                                   bg='#4CAF50', fg='white', padx=20, pady=10,
                                   command=self.run_algorithms)
        self.run_button.pack(side='left', padx=10)
        
        self.stop_button = tk.Button(button_frame, text="Stop", font=('Arial', 12, 'bold'),
                                    bg='#f44336', fg='white', padx=20, pady=10,
                                    command=self.stop_algorithms, state='disabled')
        self.stop_button.pack(side='left', padx=10)
        
        self.save_button = tk.Button(button_frame, text="Save Results", font=('Arial', 12, 'bold'),
                                    bg='#2196F3', fg='white', padx=20, pady=10,
                                    command=self.save_results)
        self.save_button.pack(side='left', padx=10)
        
        # Results area
        results_frame = tk.LabelFrame(self.root, text="Results", font=('Arial', 12, 'bold'),
                                        bg='#f0f0f0', height=500)
        results_frame.pack(pady=10, padx=20, fill='both', expand=True)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Progress and status
        self.progress_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.progress_frame.pack(fill='x', padx=20, pady=(0,10))
        
        self.progress_var = tk.StringVar(value="Ready to run")
        self.progress_label = tk.Label(self.progress_frame, textvariable=self.progress_var,
                                      font=('Arial', 10), bg='#f0f0f0')
        self.progress_label.pack(side='left')
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate')
        self.progress_bar.pack(side='right', fill='x', expand=True, padx=(10,0))
        
    def on_square_type_change(self):
        """Update size options based on square type"""
        if self.square_type.get() == "perfect":
            # Perfect magic squares only for multiples of 4
            size_combo = self.root.nametowidget(str(self.root.focus_get()))
            if hasattr(self, 'size_combo'):
                self.size_combo['values'] = ["4", "8", "12"]
                self.size_var.set("4")
        else:
            if hasattr(self, 'size_combo'):
                self.size_combo['values'] = ["3", "4", "5", "6", "7", "8"]
                self.size_var.set("3")
    
    def run_algorithms(self):
        """Run selected algorithms"""
        if self.is_running:
            return
            
        # Validate inputs
        try:
            size = int(self.size_var.get())
            pop_size = int(self.pop_size_var.get())
            max_gen = int(self.max_gen_var.get())
            mutation_rate = float(self.mutation_rate_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values")
            return
        
        if not (self.run_classic.get() or self.run_darwinian.get() or self.run_lamarckian.get()):
            messagebox.showerror("Error", "Please select at least one algorithm")
            return
        
        # Clear previous results
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        
        self.results = {}
        
        # Start algorithms in separate thread
        self.is_running = True
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_bar.start()
        
        thread = threading.Thread(target=self.run_algorithms_thread,
                                args=(size, pop_size, max_gen, mutation_rate))
        thread.daemon = True
        thread.start()
    
    def run_algorithms_thread(self, size, pop_size, max_gen, mutation_rate):
        """Run algorithms in separate thread"""
        try:
            # Create appropriate solver
            if self.square_type.get() == "standard":
                solver_class = StandardMagicSquare
            else:
                solver_class = PerfectMagicSquare
            
            algorithms = []
            if self.run_classic.get():
                algorithms.append(("Classic", "classic"))
            if self.run_darwinian.get():
                algorithms.append(("Darwinian", "darwinian"))
            if self.run_lamarckian.get():
                algorithms.append(("Lamarckian", "lamarckian"))
            
            for name, algo_type in algorithms:
                if not self.is_running:
                    break
                    
                self.progress_var.set(f"Running {name} Algorithm...")
                
                solver = solver_class(size, pop_size, max_gen, mutation_rate)
                result = solver.run(algo_type)
                
                self.results[name] = result
                
                # Update GUI with results
                self.root.after(0, self.display_result, name, result)
            
            self.root.after(0, self.algorithms_finished)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, self.algorithms_finished)
    
    def display_result(self, name, result):
        """Display results in a new tab"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text=name)
        
        # Create paned window for results
        paned = tk.PanedWindow(tab_frame, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left panel - solution and stats
        left_panel = tk.Frame(paned, bg='white')
        paned.add(left_panel, width=400)
        
        # Solution display
        solution_frame = tk.LabelFrame(left_panel, text="Best Solution", font=('Arial', 10, 'bold'))
        solution_frame.pack(fill='x', padx=5, pady=5)
        
        if result['best_solution'] is not None:
            solution_text = tk.Text(solution_frame, height=8, width=30, font=('Courier', 10))
            solution_text.pack(padx=5, pady=5)
            
            # Format and display the magic square
            square = result['best_solution']
            for row in square:
                line = ' '.join(f'{num:3d}' for num in row)
                solution_text.insert(tk.END, line + '\n')
            
            solution_text.config(state='disabled')
        else:
            tk.Label(solution_frame, text="No valid solution found", 
                    font=('Arial', 10), fg='red').pack(pady=10)
        
        # Statistics
        stats_frame = tk.LabelFrame(left_panel, text="Statistics", font=('Arial', 10, 'bold'))
        stats_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        stats_text = tk.Text(stats_frame, height=10, width=30, font=('Arial', 9))
        stats_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        stats_info = f"""Best Fitness: {result['best_fitness']:.4f}
Final Average Fitness: {result['final_avg_fitness']:.4f}
Generations Run: {result['generations_run']}
Function Evaluations: {result['function_evaluations']}
Execution Time: {result['execution_time']:.2f} seconds
Convergence Generation: {result.get('convergence_generation', 'N/A')}
Success: {'Yes' if result['best_fitness'] == 0 else 'No'}"""
        
        stats_text.insert(tk.END, stats_info)
        stats_text.config(state='disabled')
        
        # Right panel - fitness plot
        right_panel = tk.Frame(paned, bg='white')
        paned.add(right_panel)
        
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        fig.suptitle(f'{name} Algorithm - Fitness Evolution')
        
        generations = range(len(result['best_fitness_history']))
        
        # Best fitness plot
        ax1.plot(generations, result['best_fitness_history'], 'b-', linewidth=2, label='Best Fitness')
        ax1.set_ylabel('Best Fitness')
        ax1.set_title('Best Fitness Over Generations')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Average fitness plot
        ax2.plot(generations, result['avg_fitness_history'], 'r-', linewidth=2, label='Average Fitness')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Average Fitness')
        ax2.set_title('Average Fitness Over Generations')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, right_panel)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
    
    def stop_algorithms(self):
        """Stop running algorithms"""
        self.is_running = False
        
    def algorithms_finished(self):
        """Called when algorithms finish"""
        self.is_running = False
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.progress_bar.stop()
        self.progress_var.set("Algorithms completed")
        
    def save_results(self):
        """Save results to files"""
        if not self.results:
            messagebox.showwarning("Warning", "No results to save")
            return
        
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save summary report
            with open(f'magic_square_results_{timestamp}.txt', 'w') as f:
                f.write("Magic Square Genetic Algorithm Results\n")
                f.write("="*50 + "\n\n")
                
                f.write(f"Square Type: {self.square_type.get().title()}\n")
                f.write(f"Size: {self.size_var.get()}x{self.size_var.get()}\n")
                f.write(f"Population Size: {self.pop_size_var.get()}\n")
                f.write(f"Max Generations: {self.max_gen_var.get()}\n")
                f.write(f"Mutation Rate: {self.mutation_rate_var.get()}\n\n")
                
                for name, result in self.results.items():
                    f.write(f"{name} Algorithm Results:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Best Fitness: {result['best_fitness']:.6f}\n")
                    f.write(f"Final Average Fitness: {result['final_avg_fitness']:.6f}\n")
                    f.write(f"Generations Run: {result['generations_run']}\n")
                    f.write(f"Function Evaluations: {result['function_evaluations']}\n")
                    f.write(f"Execution Time: {result['execution_time']:.2f} seconds\n")
                    f.write(f"Success: {'Yes' if result['best_fitness'] == 0 else 'No'}\n")
                    
                    if result['best_solution'] is not None:
                        f.write("\nBest Solution:\n")
                        for row in result['best_solution']:
                            f.write(' '.join(f'{num:3d}' for num in row) + '\n')
                    
                    f.write("\n" + "="*50 + "\n\n")
            
            messagebox.showinfo("Success", f"Results saved to magic_square_results_{timestamp}.txt")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = MagicSquareGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()