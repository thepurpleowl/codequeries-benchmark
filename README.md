# CodeQueries Benchmark

The repo provides scripts to reproduce the important results of [Learning to Answer Semantic Queries over Code](https://arxiv.org/abs/2209.08372). The scripts can be used get performance metric on proposed approach for the CodeQueries Benchmark.

More details on the curated dataset for this benchmark are available on [huggingface](https://huggingface.co/datasets/thepurpleowl/codequeries).

### Steps
-----------
1. Clone the repo in a virtual environment.
2. Run `setup.sh` to setup the workspace.
3. Run the following commands to get performance metric values.   
**For span prediction**  
`python evaluate_spanprediction.py --example_types_to_evaluate=<positive/negative/all> --setting=<ideal/file_ideal/prefix/sliding_window/twostep>`  
**For relevance prediction**  
`python evaluate_relevance.py`


### LLM experiment
python generate_spans.py --Target_folder="<test_dir_file_fewshot_sf>" --prompt_template="prompt_templates/span_highlight_fewshot_sf.j2" --few_shot=True --with_sf=True --stop='```\nEND'

#### LLM experiment Evaluation
python evaluate_generated_spans.py --g=test_dir_file_0shot/logs
python evaluate_generated_spans.py --g=test_dir_file_fewshot_sf/logs --with_sf=True

### Benchmark
-----------
##### Span prediction results

| Model name (setting)          | All          | Positive     | Negative     |
|-------------------------------|--------------|--------------|--------------|
| CuBERT-1K (Ideal)             | 86.70        |72.51         | 96.79        |
| CuBERT-1K (Prefix)            | 72.28        |36.60         | 93.80        |
| CuBERT-1K (Sliding window)    | 73.03        |51.91         | 85.76        |
| CuBERT-1K (Two-step)          | 80.13        |52.61         | 96.73        |
| CuBERT-1K (File-level ideal)  | 82.47        |59.60         | 96.26        |



 ##### Relevance prediction results
 
| Model name                  | Precision     | Recall       | Accuracy     |
|-----------------------------|---------------|--------------|--------------|
| Relevance Classification    | 95.73         | 90.10        | 96.38        |

##### BibTeX entry and citation info
-----------
```
@misc{https://doi.org/10.48550/arxiv.2209.08372,
  doi = {10.48550/ARXIV.2209.08372},  
  url = {https://arxiv.org/abs/2209.08372},  
  author = {Sahu, Surya Prakash and Mandal, Madhurima and Bharadwaj, Shikhar and Kanade, Aditya and Maniatis, Petros and Shevade, Shirish},  
  keywords = {Software Engineering (cs.SE), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},  
  title = {Learning to Answer Semantic Queries over Code},  
  publisher = {arXiv},  
  year = {2022},  
  copyright = {Creative Commons Attribution 4.0 International}
}
```
