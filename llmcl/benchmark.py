#!/usr/bin/env python3
"""
LLMCL Benchmark Suite

Tests LLMCL against all puzzles in the puzzles folder, running each puzzle
multiple times to evaluate consistency and success rate.

Usage:
    python benchmark.py [--model gpt-4o] [--runs 3] [--force-derive] [--output results.csv]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

class BenchmarkResult:
    def __init__(self, puzzle_name: str, run_id: int):
        self.puzzle_name = puzzle_name
        self.run_id = run_id
        self.success = False
        self.num_solutions = 0
        self.error_type = None
        self.error_message = None
        self.execution_time = 0.0
        self.phases_completed = []
        self.raw_output = ""
        self.asp_file_path = None
        self.solution_text_path = None
        self.clingo_output_path = None
        self.is_perfect = False  # Exactly 1 solution

class BenchmarkSuite:
    def __init__(self, models: List[str] = None, runs_per_puzzle: int = 3, 
                 force_derive: bool = False, verbose: bool = False, output_file: str = None,
                 max_workers: int = 4):
        self.models = models or ["gpt-4o"]
        self.runs_per_puzzle = runs_per_puzzle
        self.force_derive = force_derive
        self.verbose = verbose
        self.max_workers = max_workers
        self.results: Dict[str, List[BenchmarkResult]] = {model: [] for model in self.models}
        self.puzzles_dir = Path("../puzzles")
        self.output_file = output_file
        self.completed_puzzles = {model: 0 for model in self.models}
        self.completed_runs = 0
        self.total_runs = 0
        self.start_time = None
        
        # Thread safety
        self.results_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.csv_lock = threading.Lock()
        
        # Create organized directory structure for results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_force_derive" if force_derive else "_standard"
        models_str = "_vs_".join([model.replace('-', '_') for model in self.models])
        self.results_dir = Path(f"benchmark_comparison_{models_str}{mode_suffix}_{timestamp}")
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each model and comparison
        for model in self.models:
            model_dir = self.results_dir / f"model_{model.replace('-', '_')}"
            model_dir.mkdir(exist_ok=True)
            (model_dir / "asp_files").mkdir(exist_ok=True)
            (model_dir / "solution_texts").mkdir(exist_ok=True)
            (model_dir / "clingo_outputs").mkdir(exist_ok=True)
            (model_dir / "reports").mkdir(exist_ok=True)
        
        # Create comparison directory
        (self.results_dir / "comparison").mkdir(exist_ok=True)
        
        # Initialize CSV file with headers
        if self.output_file:
            self._initialize_csv()
    
    def discover_puzzles(self) -> List[Path]:
        """Discover all JSON puzzle files in the puzzles directory."""
        if not self.puzzles_dir.exists():
            raise FileNotFoundError(f"Puzzles directory not found: {self.puzzles_dir}")
        
        puzzle_files = list(self.puzzles_dir.glob("*.json"))
        puzzle_files.sort()  # Consistent ordering
        
        print(f"📁 Found {len(puzzle_files)} puzzles in {self.puzzles_dir}")
        return puzzle_files
    
    def _initialize_csv(self) -> None:
        """Initialize CSV file with headers."""
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'model', 'puzzle_name', 'run_id', 'success', 'num_solutions', 'is_perfect',
                'error_type', 'error_message', 'execution_time', 
                'phases_completed', 'asp_file_path', 'solution_text_path', 
                'clingo_output_path', 'timestamp'
            ])
    
    def _save_result_immediately(self, result: BenchmarkResult, model: str) -> None:
        """Save a single result to CSV immediately (thread-safe)."""
        if not self.output_file:
            return
        
        with self.csv_lock:
            with open(self.output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    model,
                    result.puzzle_name,
                    result.run_id,
                    result.success,
                    result.num_solutions,
                    result.is_perfect,
                    result.error_type or '',
                    result.error_message or '',
                    f"{result.execution_time:.2f}",
                    ';'.join(result.phases_completed),
                    result.asp_file_path or '',
                    result.solution_text_path or '',
                    result.clingo_output_path or '',
                    datetime.now().isoformat()
                ])
    
    def _print_running_stats(self, puzzle_name: str, model: str = None) -> None:
        """Print running statistics after completing a puzzle."""
        all_results = []
        for model_results in self.results.values():
            all_results.extend(model_results)
        
        if not all_results:
            return
        
        # Calculate current stats
        if len(self.models) == 1:
            # Single model mode
            stats = self.generate_statistics()
            total_puzzles = stats['total_puzzles']
            completed = self.completed_puzzles[self.models[0]]
            
            print(f"\n📊 RUNNING STATISTICS (after {puzzle_name}):")
            print(f"   Model: {self.models[0]}")
        else:
            # Multi-model mode - show comparison
            print(f"\n📊 RUNNING STATISTICS (after {puzzle_name}):")
            print(f"   Models: {', '.join(self.models)}")
            
            for model in self.models:
                model_stats = self.generate_model_statistics(model)
                completed = self.completed_puzzles[model]
                total_puzzles = model_stats['total_puzzles']
                
                print(f"   {model}: {completed}/{total_puzzles} puzzles, "
                      f"{model_stats['perfect_run_rate']:.1f}% perfect, "
                      f"{model_stats['success_rate']:.1f}% success")
            
            # Overall comparison
            total_completed = sum(self.completed_puzzles.values())
            total_possible = len(self.models) * len(self.discover_puzzles())
            print(f"   Overall progress: {total_completed}/{total_possible} "
                  f"({total_completed/total_possible*100:.1f}%)")
            return
        
        # Single model detailed stats
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        if completed > 0:
            avg_time_per_puzzle = elapsed_time / completed
            remaining_puzzles = total_puzzles - completed
            est_remaining = avg_time_per_puzzle * remaining_puzzles
            est_remaining_str = f"{est_remaining/60:.1f}min" if est_remaining > 60 else f"{est_remaining:.0f}s"
        else:
            est_remaining_str = "unknown"
        
        print(f"   Progress: {completed}/{total_puzzles} puzzles ({completed/total_puzzles*100:.1f}%)")
        print(f"   Overall success rate: {stats['success_rate']:.1f}%")
        print(f"   Perfect runs (1 solution): {stats['perfect_runs']}/{stats['total_runs']} ({stats['perfect_run_rate']:.1f}%)")
        print(f"   Perfect puzzles: {stats['perfect_puzzles']}/{completed}")
        print(f"   Average time per run: {stats['avg_execution_time']:.1f}s")
        print(f"   Estimated time remaining: {est_remaining_str}")
        print(f"   📁 Results saved in: {self.results_dir}")
        
        # Show top error types if any
        if stats['error_types']:
            top_errors = sorted(stats['error_types'].items(), 
                              key=lambda x: x[1], reverse=True)[:3]
            print(f"   Top errors: {', '.join(f'{e}({c})' for e, c in top_errors)}")
        
        # Save intermediate report every 5 puzzles
        if self.output_file and completed % 5 == 0:
            report_file = self.output_file.replace('.csv', f'_intermediate_report_{model}_{completed}.txt')
            self._save_intermediate_report(report_file, model)
            print(f"   📄 Intermediate report saved: {report_file}")
        
        print("   " + "="*50)
    
    def generate_model_statistics(self, model: str) -> Dict:
        """Generate statistics for a specific model."""
        model_results = self.results.get(model, [])
        if not model_results:
            return {'total_runs': 0, 'total_puzzles': 0, 'successful_runs': 0, 
                   'success_rate': 0, 'perfect_runs': 0, 'perfect_run_rate': 0,
                   'perfect_puzzles': 0, 'perfect_puzzle_rate': 0}
        
        total_runs = len(model_results)
        successful_runs = sum(1 for r in model_results if r.success)
        perfect_runs = sum(1 for r in model_results if r.is_perfect)
        
        # Group by puzzle
        puzzles = {}
        for result in model_results:
            if result.puzzle_name not in puzzles:
                puzzles[result.puzzle_name] = []
            puzzles[result.puzzle_name].append(result)
        
        perfect_puzzles = 0
        for puzzle_name, puzzle_results in puzzles.items():
            if all(r.is_perfect for r in puzzle_results if r.success):
                perfect_puzzles += 1
        
        execution_times = [r.execution_time for r in model_results]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            'total_runs': total_runs,
            'total_puzzles': len(puzzles),
            'successful_runs': successful_runs,
            'success_rate': successful_runs / total_runs * 100 if total_runs > 0 else 0,
            'perfect_runs': perfect_runs,
            'perfect_run_rate': perfect_runs / total_runs * 100 if total_runs > 0 else 0,
            'perfect_puzzles': perfect_puzzles,
            'perfect_puzzle_rate': perfect_puzzles / len(puzzles) * 100 if puzzles else 0,
            'avg_execution_time': avg_execution_time
        }
    
    def _save_intermediate_report(self, report_file: str, model: str = None) -> None:
        """Save an intermediate text report."""
        stats = self.generate_statistics()
        
        with open(report_file, 'w') as f:
            f.write(f"LLMCL BENCHMARK - INTERMEDIATE REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Models: {', '.join(self.models)}\n")
            f.write(f"  Force-derive: {self.force_derive}\n")
            f.write(f"  Runs per puzzle: {self.runs_per_puzzle}\n\n")
            
            f.write(f"Progress:\n")
            f.write(f"  Completed puzzles: {sum(self.completed_puzzles.values())}/{stats['total_puzzles']}\n")
            f.write(f"  Total runs: {stats['total_runs']}\n")
            f.write(f"  Success rate: {stats['success_rate']:.1f}%\n")
            f.write(f"  Perfect puzzles: {stats['perfect_puzzles']}/{sum(self.completed_puzzles.values())}\n\n")
            
            if stats['error_types']:
                f.write("Error breakdown:\n")
                for error_type, count in sorted(stats['error_types'].items(), 
                                              key=lambda x: x[1], reverse=True):
                    f.write(f"  {error_type}: {count}\n")
                f.write("\n")
            
            # Puzzle-by-puzzle results
            puzzles = {}
            for model_results in self.results.values():
                for result in model_results:
                    if result.puzzle_name not in puzzles:
                        puzzles[result.puzzle_name] = []
                    puzzles[result.puzzle_name].append(result)
            
            f.write("Puzzle results:\n")
            for puzzle_name in sorted(puzzles.keys()):
                puzzle_results = puzzles[puzzle_name]
                successful = sum(1 for r in puzzle_results if r.success)
                f.write(f"  {puzzle_name}: {successful}/{len(puzzle_results)} success\n")
    
    def run_single_puzzle(self, puzzle_file: Path, run_id: int, model: str) -> BenchmarkResult:
        """Run LLMCL on a single puzzle and capture results."""
        result = BenchmarkResult(puzzle_file.stem, run_id)
        
        # Create file paths for this run
        run_prefix = f"{puzzle_file.stem}_run{run_id}"
        model_dir = self.results_dir / f"model_{model.replace('-', '_')}"
        asp_file = model_dir / "asp_files" / f"{run_prefix}.lp"
        solution_file = model_dir / "solution_texts" / f"{run_prefix}_solution.txt"
        clingo_file = model_dir / "clingo_outputs" / f"{run_prefix}_clingo.txt"
        
        # Store paths in result (use absolute paths to avoid path resolution issues)
        result.asp_file_path = str(asp_file.absolute())
        result.solution_text_path = str(solution_file.absolute())
        result.clingo_output_path = str(clingo_file.absolute())
        
        # Build command
        cmd = [
            sys.executable, "-m", "llmcl",
            str(puzzle_file),
            "--model", model,
            "--output", str(asp_file)
        ]
        
        if self.force_derive:
            cmd.append("--force-derive")
        
        if self.verbose:
            print(f"      Running: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # Run LLMCL
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            result.execution_time = time.time() - start_time
            result.raw_output = proc.stdout + "\n" + proc.stderr
            
            # Save the complete output for analysis
            with open(clingo_file, 'w') as f:
                f.write("LLMCL OUTPUT:\n")
                f.write("=" * 50 + "\n")
                f.write(result.raw_output)
                f.write("\n" + "=" * 50 + "\n")
            
            # Parse output for success/failure
            if proc.returncode == 0:
                # Check if puzzle was solved successfully
                if "🎉 PUZZLE SOLVED!" in result.raw_output:
                    result.success = True
                    
                    # Extract number of solutions
                    solutions_match = re.search(r"🎯 FOUND (\d+) SOLUTION\(S\)", result.raw_output)
                    if solutions_match:
                        result.num_solutions = int(solutions_match.group(1))
                    else:
                        result.num_solutions = 1  # Assume 1 if not specified
                    
                    # Check if it's a perfect solution (exactly 1)
                    result.is_perfect = (result.num_solutions == 1)
                    result.success = result.is_perfect  # Only 1 solution counts as success
                    
                    # Categorize solution count issues
                    if result.num_solutions == 0:
                        result.error_type = "no_solutions"
                        result.error_message = "ASP generated but no solutions found (over-constrained)"
                        result.success = False
                    elif result.num_solutions > 1:
                        result.error_type = "multiple_solutions" 
                        result.error_message = f"Found {result.num_solutions} solutions (under-constrained)"
                        result.success = False
                    
                    # Extract and save formatted solution text
                    solution_start = result.raw_output.find("🔸 SOLUTION 1:")
                    if solution_start != -1:
                        # Find the end of the solutions section
                        raw_output_start = result.raw_output.find("🔍 Raw clingo output:")
                        if raw_output_start != -1:
                            solution_text = result.raw_output[solution_start:raw_output_start].strip()
                        else:
                            solution_text = result.raw_output[solution_start:].strip()
                        
                        with open(solution_file, 'w') as f:
                            f.write(f"PUZZLE: {puzzle_file.name}\n")
                            f.write(f"RUN: {run_id}\n")
                            f.write(f"SOLUTIONS FOUND: {result.num_solutions}\n")
                            f.write(f"PERFECT: {result.is_perfect}\n")
                            f.write("=" * 60 + "\n\n")
                            f.write(solution_text)
                    
                    # Extract and append raw clingo output
                    raw_clingo_start = result.raw_output.find("🔍 Raw clingo output:")
                    if raw_clingo_start != -1:
                        raw_clingo = result.raw_output[raw_clingo_start:]
                        with open(clingo_file, 'a') as f:
                            f.write("\nRAW CLINGO OUTPUT:\n")
                            f.write("=" * 50 + "\n")
                            f.write(raw_clingo)
                else:
                    # Generated ASP but didn't solve
                    result.error_type = "solving_failed"
                    if "❌ Solving failed:" in result.raw_output:
                        error_line = next((line for line in result.raw_output.split('\n') 
                                         if "❌ Solving failed:" in line), "")
                        result.error_message = error_line.replace("❌ Solving failed:", "").strip()
                    else:
                        result.error_message = "Unknown solving failure"
            else:
                # Process failed
                result.success = False
                
                # Categorize error type
                if "Error extracting puzzle structure:" in result.raw_output:
                    result.error_type = "extraction_error"
                elif "Error generating ASP facts:" in result.raw_output:
                    result.error_type = "asp_generation_error"
                elif "ASP syntax validation failed:" in result.raw_output:
                    result.error_type = "syntax_error"
                elif "Clingo syntax errors detected:" in result.raw_output:
                    result.error_type = "clingo_error"
                elif "timeout" in result.raw_output.lower():
                    result.error_type = "timeout"
                elif "API" in result.raw_output and "key" in result.raw_output:
                    result.error_type = "api_key_error"
                else:
                    result.error_type = "unknown_error"
                
                # Extract error message
                error_lines = [line for line in result.raw_output.split('\n') 
                             if "❌" in line or "Error" in line]
                if error_lines:
                    result.error_message = error_lines[0].strip()
                else:
                    result.error_message = f"Return code: {proc.returncode}"
            
            # Extract completed phases
            phase_patterns = [
                r"✅ Successfully extracted puzzle structure",
                r"✅ Successfully generated \d+ ASP facts",
                r"✅ ASP syntax validation passed",
                r"🎉 PUZZLE SOLVED!"
            ]
            
            for i, pattern in enumerate(phase_patterns, 1):
                if re.search(pattern, result.raw_output):
                    result.phases_completed.append(f"Phase {i}")
        
        except subprocess.TimeoutExpired:
            result.execution_time = time.time() - start_time
            result.error_type = "timeout"
            result.error_message = "Execution timeout (5 minutes)"
            
            # Still save what we have
            with open(clingo_file, 'w') as f:
                f.write("TIMEOUT OCCURRED\n")
                f.write(f"Execution time: {result.execution_time:.2f}s\n")
        
        except Exception as e:
            result.execution_time = time.time() - start_time
            result.error_type = "execution_error"
            result.error_message = str(e)
            
            # Save error information
            with open(clingo_file, 'w') as f:
                f.write(f"EXECUTION ERROR: {e}\n")
                f.write(f"Execution time: {result.execution_time:.2f}s\n")
        
        return result
    
    def _run_puzzle_model_combination(self, puzzle_file: Path, model: str, run_id: int) -> Tuple[BenchmarkResult, str, Path]:
        """Run a single puzzle-model-run combination (for parallel execution)."""
        result = self.run_single_puzzle(puzzle_file, run_id, model)
        
        # Thread-safe result storage
        with self.results_lock:
            self.results[model].append(result)
        
        # Thread-safe progress tracking
        with self.progress_lock:
            self.completed_runs += 1
            progress = (self.completed_runs / self.total_runs) * 100
            
            # Show immediate result
            if result.success:
                progress_msg = f"[{progress:.1f}%] 🎯 {model} | {puzzle_file.name} | Run {run_id}: 1 solution in {result.execution_time:.1f}s"
            else:
                error_indicator = "❌"
                if result.error_type == "multiple_solutions":
                    progress_msg = f"[{progress:.1f}%] {error_indicator} {model} | {puzzle_file.name} | Run {run_id}: {result.num_solutions} solutions (under-constrained) in {result.execution_time:.1f}s"
                elif result.error_type == "no_solutions":
                    progress_msg = f"[{progress:.1f}%] {error_indicator} {model} | {puzzle_file.name} | Run {run_id}: 0 solutions (over-constrained) in {result.execution_time:.1f}s"
                else:
                    progress_msg = f"[{progress:.1f}%] {error_indicator} {model} | {puzzle_file.name} | Run {run_id}: {result.error_type} in {result.execution_time:.1f}s"
            
            print(progress_msg)
            
            if self.verbose and result.error_message and not result.success:
                print(f"         {result.error_message}")
        
        # Save result immediately
        self._save_result_immediately(result, model)
        
        return result, model, puzzle_file
    
    def _print_periodic_stats(self) -> None:
        """Print periodic statistics during parallel execution (thread-safe)."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        progress = (self.completed_runs / self.total_runs) * 100
        
        # Estimate time remaining
        if self.completed_runs > 0:
            avg_time_per_run = elapsed_time / self.completed_runs
            remaining_runs = self.total_runs - self.completed_runs
            est_remaining = avg_time_per_run * remaining_runs
            est_remaining_str = f"{est_remaining/60:.1f}min" if est_remaining > 60 else f"{est_remaining:.0f}s"
        else:
            est_remaining_str = "unknown"
        
        # Calculate current success rates for each model
        print(f"\n📊 PROGRESS UPDATE [{progress:.1f}%] - {self.completed_runs}/{self.total_runs} runs")
        print(f"⏱️  Elapsed: {elapsed_time/60:.1f}min | Est. remaining: {est_remaining_str}")
        
        if len(self.models) > 1:
            print("🤖 Per-model progress:")
            for model in self.models:
                model_results = self.results[model]
                if model_results:
                    success_count = sum(1 for r in model_results if r.success)
                    multiple_count = sum(1 for r in model_results if r.error_type == "multiple_solutions")
                    no_solutions_count = sum(1 for r in model_results if r.error_type == "no_solutions")
                    other_failures = len(model_results) - success_count - multiple_count - no_solutions_count
                    total_count = len(model_results)
                    success_rate = success_count / total_count * 100 if total_count > 0 else 0
                    
                    print(f"   {model}: {total_count} runs | {success_rate:.1f}% success (1 solution)")
                    if multiple_count > 0 or no_solutions_count > 0 or other_failures > 0:
                        failure_details = []
                        if multiple_count > 0:
                            failure_details.append(f"{multiple_count} multi-sol")
                        if no_solutions_count > 0:
                            failure_details.append(f"{no_solutions_count} no-sol")
                        if other_failures > 0:
                            failure_details.append(f"{other_failures} other")
                        print(f"      Failures: {', '.join(failure_details)}")
                else:
                    print(f"   {model}: 0 runs")
        else:
            # Single model detailed stats
            model = self.models[0]
            model_results = self.results[model]
            if model_results:
                success_count = sum(1 for r in model_results if r.success)
                multiple_count = sum(1 for r in model_results if r.error_type == "multiple_solutions")
                no_solutions_count = sum(1 for r in model_results if r.error_type == "no_solutions")
                total_count = len(model_results)
                success_rate = success_count / total_count * 100 if total_count > 0 else 0
                
                print(f"🎯 {model}: {success_count}/{total_count} success ({success_rate:.1f}%) - exactly 1 solution")
                if multiple_count > 0:
                    print(f"   Under-constrained: {multiple_count} runs with multiple solutions")
                if no_solutions_count > 0:
                    print(f"   Over-constrained: {no_solutions_count} runs with no solutions")
        
        print("=" * 60)
    
    def run_benchmark(self) -> None:
        """Run the full benchmark suite with parallel execution."""
        puzzles = self.discover_puzzles()
        total_runs_per_model = len(puzzles) * self.runs_per_puzzle
        self.total_runs = total_runs_per_model * len(self.models)
        
        print(f"\n🚀 Starting LLMCL Multi-Model Benchmark (Parallel)")
        print(f"📊 Models: {', '.join(self.models)}")
        print(f"🔄 Runs per puzzle: {self.runs_per_puzzle}")
        print(f"🧠 Force-derive: {self.force_derive}")
        print(f"⚡ Max parallel workers: {self.max_workers}")
        print(f"📋 Total runs: {self.total_runs} ({total_runs_per_model} per model)")
        if self.output_file:
            print(f"💾 Continuous saving to: {self.output_file}")
        print("=" * 80)
        
        # Create all tasks (puzzle-model-run combinations)
        tasks = []
        for model in self.models:
            for puzzle_file in puzzles:
                for run_id in range(1, self.runs_per_puzzle + 1):
                    tasks.append((puzzle_file, model, run_id))
        
        print(f"\n🎯 Created {len(tasks)} tasks for parallel execution")
        
        # Start timing
        self.start_time = time.time()
        self.completed_runs = 0
        
        # Early failure detection variables
        early_check_threshold = min(10, len(tasks) // 4)  # Check after 10 tasks or 25% completion
        early_failure_threshold = 0.9  # If 90% of early tasks fail with same error, abort
        
        # Run tasks in parallel
        completed_puzzles_per_model = {model: set() for model in self.models}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._run_puzzle_model_combination, puzzle_file, model, run_id): (puzzle_file, model, run_id)
                for puzzle_file, model, run_id in tasks
            }
            
            print(f"🚀 Submitted {len(future_to_task)} tasks to {self.max_workers} workers")
            print("📊 Progress will be shown as tasks complete...\n")
            
            # Process completed tasks
            early_results = []
            should_abort = False
            
            for future in as_completed(future_to_task):
                puzzle_file, model, run_id = future_to_task[future]
                
                try:
                    result, model, puzzle_file = future.result()
                    
                    # Collect early results for failure detection
                    if len(early_results) < early_check_threshold:
                        early_results.append(result)
                    
                    # Check for early systematic failure
                    if len(early_results) == early_check_threshold and not should_abort:
                        failed_results = [r for r in early_results if not r.success]
                        if len(failed_results) >= early_failure_threshold * early_check_threshold:
                            # Check if all failures have the same error type
                            error_types = [r.error_type for r in failed_results if r.error_type]
                            if error_types and len(set(error_types)) == 1:
                                dominant_error = error_types[0]
                                failure_rate = len(failed_results) / len(early_results) * 100
                                
                                print(f"\n🚨 EARLY FAILURE DETECTION TRIGGERED!")
                                print(f"   📊 {failure_rate:.0f}% of first {len(early_results)} tasks failed with '{dominant_error}'")
                                
                                if dominant_error == "api_key_error":
                                    print(f"   🔑 API key issue detected. Please check your API configuration:")
                                    print(f"      - Set OPENAI_API_KEY environment variable for GPT models")
                                    print(f"      - Set ANTHROPIC_API_KEY environment variable for Claude models")
                                elif dominant_error == "timeout":
                                    print(f"   ⏱️  Timeout issue detected. Consider reducing parallelism with --max-workers")
                                elif dominant_error in ["extraction_error", "asp_generation_error"]:
                                    print(f"   🧩 Systematic puzzle processing issue detected")
                                
                                print(f"   🛑 Aborting benchmark to avoid wasting resources...")
                                should_abort = True
                                
                                # Cancel remaining futures
                                for remaining_future in future_to_task:
                                    if not remaining_future.done():
                                        remaining_future.cancel()
                                break
                    
                    # Normal processing continues if no early abort
                    if not should_abort:
                        # Track completed puzzles for each model
                        puzzle_name = puzzle_file.stem
                        completed_puzzles_per_model[model].add(puzzle_name)
                        
                        # Update completed puzzles count
                        with self.progress_lock:
                            self.completed_puzzles[model] = len(completed_puzzles_per_model[model])
                        
                        # Show periodic statistics (every 10% progress)
                        current_progress = (self.completed_runs / self.total_runs) * 100
                        if self.completed_runs % max(1, self.total_runs // 10) == 0:
                            self._print_periodic_stats()
                
                except Exception as e:
                    print(f"❌ Task failed: {puzzle_file.name} | {model} | Run {run_id}: {e}")
                    if self.verbose:
                        import traceback
                        traceback.print_exc()
        
        if should_abort:
            print(f"\n🛑 Benchmark aborted due to systematic failures")
            print(f"📊 Completed {self.completed_runs}/{self.total_runs} tasks before aborting")
            print(f"💡 Fix the configuration issue and try again")
        else:
            print("\n" + "=" * 80)
            print("🏁 Parallel Multi-Model Benchmark completed!")
        
        # Generate final comparison only if we have meaningful results
        if not should_abort and len(self.models) > 1:
            self._generate_model_comparison()
    
    def generate_statistics(self) -> Dict:
        """Generate comprehensive statistics from benchmark results."""
        # Combine all results from all models for overall stats
        all_results = []
        for model_results in self.results.values():
            all_results.extend(model_results)
        
        if not all_results:
            return {'total_runs': 0, 'total_puzzles': 0, 'successful_runs': 0, 
                   'success_rate': 0, 'perfect_runs': 0, 'perfect_run_rate': 0,
                   'perfect_puzzles': 0, 'perfect_puzzle_rate': 0}
        
        total_runs = len(all_results)
        successful_runs = sum(1 for r in all_results if r.success)
        
        # Group by puzzle
        puzzles = {}
        for result in all_results:
            if result.puzzle_name not in puzzles:
                puzzles[result.puzzle_name] = []
            puzzles[result.puzzle_name].append(result)
        
        # Calculate puzzle-level statistics
        perfect_puzzles = 0  # All runs successful with exactly 1 solution
        consistent_puzzles = 0  # All runs have same result
        partial_success_puzzles = 0  # Some runs successful
        
        solution_counts = []
        execution_times = []
        error_types = {}
        
        # Count different failure types
        no_solution_runs = 0
        multiple_solution_runs = 0
        other_error_runs = 0
        
        for puzzle_name, puzzle_results in puzzles.items():
            successful_puzzle_runs = [r for r in puzzle_results if r.success]  # Only 1-solution runs
            
            # Check if all runs successful with 1 solution
            if len(successful_puzzle_runs) == len(puzzle_results):
                perfect_puzzles += 1
            
            # Check consistency
            if len(set((r.success, r.num_solutions, r.error_type) for r in puzzle_results)) == 1:
                consistent_puzzles += 1
            
            # Partial success
            if 0 < len(successful_puzzle_runs) < len(puzzle_results):
                partial_success_puzzles += 1
            
            # Collect metrics
            for result in puzzle_results:
                if result.success:
                    solution_counts.append(result.num_solutions)  # Should always be 1
                execution_times.append(result.execution_time)
                
                # Count failure types
                if result.error_type == "no_solutions":
                    no_solution_runs += 1
                elif result.error_type == "multiple_solutions":
                    multiple_solution_runs += 1
                elif result.error_type and not result.success:
                    other_error_runs += 1
                
                if result.error_type:
                    error_types[result.error_type] = error_types.get(result.error_type, 0) + 1
        
        # Calculate averages
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        avg_solutions = sum(solution_counts) / len(solution_counts) if solution_counts else 0
        
        return {
            'total_runs': total_runs,
            'total_puzzles': len(puzzles),
            'successful_runs': successful_runs,
            'success_rate': successful_runs / total_runs * 100 if total_runs > 0 else 0,
            'perfect_runs': successful_runs,  # Same as successful_runs now (only 1 solution = success)
            'perfect_run_rate': successful_runs / total_runs * 100 if total_runs > 0 else 0,
            'perfect_puzzles': perfect_puzzles,
            'perfect_puzzle_rate': perfect_puzzles / len(puzzles) * 100 if puzzles else 0,
            'consistent_puzzles': consistent_puzzles,
            'consistency_rate': consistent_puzzles / len(puzzles) * 100 if puzzles else 0,
            'partial_success_puzzles': partial_success_puzzles,
            'avg_execution_time': avg_execution_time,
            'avg_solutions': avg_solutions,
            'error_types': error_types,
            'failure_breakdown': {
                'no_solutions': no_solution_runs,
                'multiple_solutions': multiple_solution_runs,
                'other_errors': other_error_runs
            },
            'solution_distribution': {
                '0_solutions': sum(1 for c in solution_counts if c == 0),  # Should be 0 now
                '1_solution': sum(1 for c in solution_counts if c == 1),  # Should equal successful_runs
                'multiple_solutions': sum(1 for c in solution_counts if c > 1)  # Should be 0 now
            }
        }
    
    def save_results(self, output_file: str = None) -> None:
        """Save detailed results to CSV file (if not already being saved continuously)."""
        target_file = output_file or self.output_file
        
        if not target_file:
            print("⚠️ No output file specified, skipping save")
            return
        
        # If we're saving continuously, just confirm the file exists
        if self.output_file and target_file == self.output_file:
            if Path(target_file).exists():
                print(f"💾 Results continuously saved to: {target_file}")
                print(f"📊 Total entries: {len(self.results)}")
            return
        
        # Otherwise, create a new file
        with open(target_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'puzzle_name', 'run_id', 'success', 'num_solutions', 'is_perfect',
                'error_type', 'error_message', 'execution_time', 
                'phases_completed', 'asp_file_path', 'solution_text_path',
                'clingo_output_path', 'timestamp'
            ])
            
            # Data
            for model, model_results in self.results.items():
                for result in model_results:
                    writer.writerow([
                        model,
                        result.puzzle_name,
                        result.run_id,
                        result.success,
                        result.num_solutions,
                        result.is_perfect,
                        result.error_type or '',
                        result.error_message or '',
                        f"{result.execution_time:.2f}",
                        ';'.join(result.phases_completed),
                        result.asp_file_path or '',
                        result.solution_text_path or '',
                        result.clingo_output_path or '',
                        datetime.now().isoformat()
                    ])
        
        print(f"💾 Detailed results saved to: {target_file}")
    
    def print_report(self) -> None:
        """Print comprehensive benchmark report."""
        stats = self.generate_statistics()
        
        print("\n" + "=" * 60)
        print("📊 LLMCL BENCHMARK REPORT")
        print("=" * 60)
        
        print(f"\n🔧 Configuration:")
        print(f"   Models: {', '.join(self.models)}")
        print(f"   Force-derive: {self.force_derive}")
        print(f"   Runs per puzzle: {self.runs_per_puzzle}")
        
        print(f"\n📈 Overall Results:")
        print(f"   Total puzzles: {stats['total_puzzles']}")
        print(f"   Total runs: {stats['total_runs']}")
        print(f"   Successful runs (exactly 1 solution): {stats['successful_runs']}/{stats['total_runs']} "
              f"({stats['success_rate']:.1f}%)")
        
        print(f"\n🎯 Solution Quality Analysis:")
        print(f"   Perfect puzzles (all runs → 1 solution): {stats['perfect_puzzles']}/{stats['total_puzzles']} "
              f"({stats['perfect_puzzle_rate']:.1f}%)")
        print(f"   Consistent puzzles (same result all runs): {stats['consistent_puzzles']}/{stats['total_puzzles']} "
              f"({stats['consistency_rate']:.1f}%)")
        print(f"   Partially successful puzzles: {stats['partial_success_puzzles']}")
        
        # Show failure breakdown
        failure_breakdown = stats.get('failure_breakdown', {})
        if any(failure_breakdown.values()):
            print(f"\n❌ Failure Analysis:")
            if failure_breakdown.get('multiple_solutions', 0) > 0:
                print(f"   Under-constrained (multiple solutions): {failure_breakdown['multiple_solutions']} runs")
            if failure_breakdown.get('no_solutions', 0) > 0:
                print(f"   Over-constrained (no solutions): {failure_breakdown['no_solutions']} runs")
            if failure_breakdown.get('other_errors', 0) > 0:
                print(f"   Other errors (parsing/API/etc): {failure_breakdown['other_errors']} runs")
        
        print(f"\n⏱️ Performance Metrics:")
        print(f"   Average execution time: {stats['avg_execution_time']:.1f}s")
        if stats['avg_solutions'] > 0:
            print(f"   Average solutions per successful run: {stats['avg_solutions']:.1f}")
        
        print(f"\n🎲 Solution Quality Distribution:")
        sol_dist = stats['solution_distribution']
        print(f"   Exactly 1 solution (success): {sol_dist['1_solution']}")
        print(f"   Multiple solutions (under-constrained): {stats.get('failure_breakdown', {}).get('multiple_solutions', 0)}")
        print(f"   No solutions (over-constrained): {stats.get('failure_breakdown', {}).get('no_solutions', 0)}")
        print(f"   Other failures: {stats.get('failure_breakdown', {}).get('other_errors', 0)}")
        
        if stats['error_types']:
            print(f"\n❌ Error Breakdown:")
            for error_type, count in sorted(stats['error_types'].items(), 
                                          key=lambda x: x[1], reverse=True):
                print(f"   {error_type}: {count}")
        
        # Puzzle-specific summary
        print(f"\n📋 Puzzle-by-Puzzle Summary:")
        puzzles = {}
        for model_results in self.results.values():
            for result in model_results:
                if result.puzzle_name not in puzzles:
                    puzzles[result.puzzle_name] = []
                puzzles[result.puzzle_name].append(result)
        
        for puzzle_name in sorted(puzzles.keys()):
            puzzle_results = puzzles[puzzle_name]
            successful = sum(1 for r in puzzle_results if r.success)
            avg_time = sum(r.execution_time for r in puzzle_results) / len(puzzle_results)
            
            # Status symbol
            if successful == len(puzzle_results):
                if all(r.num_solutions == 1 for r in puzzle_results if r.success):
                    status = "🎯"  # Perfect
                else:
                    status = "✅"  # All successful but not ideal solutions
            elif successful > 0:
                status = "⚠️"   # Partial success
            else:
                status = "❌"   # All failed
            
            print(f"   {status} {puzzle_name}: {successful}/{len(puzzle_results)} success, "
                  f"avg {avg_time:.1f}s")
        
        print(f"\n📁 Detailed Results Structure:")
        print(f"   {self.results_dir}/")
        print(f"   ├── asp_files/          (Generated ASP code for each run)")
        print(f"   ├── solution_texts/     (Formatted puzzle solutions)")
        print(f"   ├── clingo_outputs/     (Raw clingo solver outputs)")
        print(f"   └── reports/            (Intermediate progress reports)")
        if self.output_file:
            print(f"   Main CSV: {self.output_file}")
        
        print(f"\n🏆 Key Insights:")
        success_rate = stats['success_rate']
        if success_rate >= 90:
            print(f"   🟢 Excellent performance: {success_rate:.1f}% exact solutions")
        elif success_rate >= 70:
            print(f"   🟡 Good performance: {success_rate:.1f}% exact solutions")
        elif success_rate >= 50:
            print(f"   🟠 Moderate performance: {success_rate:.1f}% exact solutions")
        else:
            print(f"   🔴 Needs improvement: {success_rate:.1f}% exact solutions")
        
        if stats['consistency_rate'] >= 80:
            print(f"   🔄 High consistency: {stats['consistency_rate']:.1f}% of puzzles give same results")
        else:
            print(f"   ⚠️ Low consistency: {stats['consistency_rate']:.1f}% of puzzles give same results")
        
        # Highlight main failure modes
        failure_breakdown = stats.get('failure_breakdown', {})
        total_failures = sum(failure_breakdown.values())
        if total_failures > 0:
            if failure_breakdown.get('multiple_solutions', 0) > failure_breakdown.get('no_solutions', 0):
                print(f"   🔧 Main issue: Under-constraining (multiple solutions) - tighten ASP rules")
            elif failure_breakdown.get('no_solutions', 0) > 0:
                print(f"   🔧 Main issue: Over-constraining (no solutions) - relax ASP rules")
            else:
                print(f"   🔧 Main issue: ASP generation errors - check puzzle parsing")
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        # We keep the results directory as it contains valuable benchmark data
        print(f"✅ Benchmark results preserved in: {self.results_dir}")
        
        # Count files across all model directories
        total_asp = 0
        total_solutions = 0
        total_clingo = 0
        
        for model in self.models:
            model_dir = self.results_dir / f"model_{model.replace('-', '_')}"
            if model_dir.exists():
                total_asp += len(list((model_dir / 'asp_files').glob('*.lp')))
                total_solutions += len(list((model_dir / 'solution_texts').glob('*.txt')))
                total_clingo += len(list((model_dir / 'clingo_outputs').glob('*.txt')))
        
        print(f"📁 Contains {total_asp} ASP files across all models")
        print(f"📁 Contains {total_solutions} solution texts across all models")
        print(f"📁 Contains {total_clingo} clingo outputs across all models")
    
    def _generate_model_comparison(self) -> None:
        """Generate detailed comparison between models."""
        print(f"\n🏆 MODEL COMPARISON REPORT")
        print("=" * 60)
        
        # Generate stats for each model
        model_stats = {}
        for model in self.models:
            model_stats[model] = self.generate_model_statistics(model)
        
        # Create comparison table
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"{'Model':<25} {'Perfect Rate':<12} {'Success Rate':<12} {'Avg Time':<10}")
        print("-" * 60)
        
        best_perfect = 0
        best_model = ""
        
        for model in self.models:
            stats = model_stats[model]
            perfect_rate = stats['perfect_run_rate']
            success_rate = stats['success_rate']
            avg_time = stats['avg_execution_time']
            
            if perfect_rate > best_perfect:
                best_perfect = perfect_rate
                best_model = model
            
            indicator = "🏆" if model == best_model else "  "
            print(f"{indicator} {model:<23} {perfect_rate:>8.1f}%    {success_rate:>8.1f}%    {avg_time:>7.1f}s")
        
        print(f"\n🎯 DETAILED COMPARISON:")
        
        # Find puzzles where models differ
        all_puzzles = set()
        for model_results in self.results.values():
            for result in model_results:
                all_puzzles.add(result.puzzle_name)
        
        disagreements = []
        perfect_counts = {model: 0 for model in self.models}
        
        for puzzle in sorted(all_puzzles):
            puzzle_results = {model: [] for model in self.models}
            
            # Collect results for this puzzle from each model
            for model in self.models:
                for result in self.results[model]:
                    if result.puzzle_name == puzzle:
                        puzzle_results[model].append(result)
            
            # Check if models agree
            model_outcomes = {}
            for model in self.models:
                results = puzzle_results[model]
                if results:
                    perfect_runs = sum(1 for r in results if r.is_perfect)
                    total_runs = len(results)
                    success_runs = sum(1 for r in results if r.success)
                    
                    if perfect_runs == total_runs:
                        model_outcomes[model] = "PERFECT"
                        perfect_counts[model] += 1
                    elif success_runs > 0:
                        model_outcomes[model] = "PARTIAL"
                    else:
                        model_outcomes[model] = "FAILED"
                else:
                    model_outcomes[model] = "NO_DATA"
            
            # Check for disagreements
            unique_outcomes = set(model_outcomes.values())
            if len(unique_outcomes) > 1:
                disagreements.append((puzzle, model_outcomes))
        
        print(f"\n🎯 Perfect Solution Counts:")
        for model in sorted(self.models, key=lambda m: perfect_counts[m], reverse=True):
            print(f"   {model}: {perfect_counts[model]}/{len(all_puzzles)} puzzles ({perfect_counts[model]/len(all_puzzles)*100:.1f}%)")
        
        if disagreements:
            print(f"\n⚡ Model Disagreements ({len(disagreements)} puzzles):")
            for puzzle, outcomes in disagreements[:10]:  # Show first 10
                print(f"   {puzzle}:")
                for model, outcome in outcomes.items():
                    print(f"     {model}: {outcome}")
            
            if len(disagreements) > 10:
                print(f"   ... and {len(disagreements) - 10} more")
        else:
            print(f"\n✅ All models agree on all puzzles!")
        
        # Save comparison report
        comparison_file = self.results_dir / "comparison" / "model_comparison_report.txt"
        with open(comparison_file, 'w') as f:
            f.write("LLMCL MODEL COMPARISON REPORT\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Models tested: {', '.join(self.models)}\n")
            f.write(f"Runs per puzzle: {self.runs_per_puzzle}\n")
            f.write(f"Force-derive mode: {self.force_derive}\n\n")
            
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 20 + "\n")
            for model in self.models:
                stats = model_stats[model]
                f.write(f"{model}:\n")
                f.write(f"  Perfect rate: {stats['perfect_run_rate']:.1f}%\n")
                f.write(f"  Success rate: {stats['success_rate']:.1f}%\n")
                f.write(f"  Avg time: {stats['avg_execution_time']:.1f}s\n\n")
            
            f.write(f"WINNER: {best_model} with {best_perfect:.1f}% perfect solutions\n")
        
        print(f"\n💾 Detailed comparison saved to: {comparison_file}")


def main():
    parser = argparse.ArgumentParser(
        description="LLMCL Benchmark Suite - Test against all puzzles",
        epilog="Examples:\n" +
               "  Single model:     python benchmark.py --model gpt-4o --runs 3 --force-derive\n" +
               "  Model comparison: python benchmark.py --model gpt-4o claude-3-5-sonnet-20241022 --runs 3\n" +
               "  High parallelism: python benchmark.py --model gpt-4o --runs 3 --max-workers 8",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        "--models",
        nargs="+",
        default=["gpt-4o"],
        help="LLM model(s) to use for benchmarking. For comparison, provide multiple models separated by spaces (default: gpt-4o)"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per puzzle (default: 3)"
    )
    
    parser.add_argument(
        "--force-derive",
        action="store_true",
        help="Use force-derive mode for all puzzles"
    )
    
    parser.add_argument(
        "--output",
        default=None,
        help="Output file for detailed results (default: benchmark_TIMESTAMP.csv)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Generate default output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_force_derive" if args.force_derive else "_standard"
        models_str = "_vs_".join([model.replace('-', '_') for model in args.model])
        args.output = f"benchmark_{models_str}{mode_suffix}_{timestamp}.csv"
    
    # Create and run benchmark
    benchmark = BenchmarkSuite(
        models=args.model,
        runs_per_puzzle=args.runs,
        force_derive=args.force_derive,
        verbose=args.verbose,
        output_file=args.output,
        max_workers=args.max_workers
    )
    
    try:
        benchmark.run_benchmark()
        benchmark.print_report()
        benchmark.save_results(args.output)
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
        if benchmark.results:
            print("📊 Generating partial report...")
            benchmark.print_report()
            benchmark.save_results(args.output)
    
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)
    
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    main() 