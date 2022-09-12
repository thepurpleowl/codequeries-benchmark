# Codequeries-Benchmark

The repo provides scripts to reproduce the important results of [preprint](arxiv link). The scripts can be used get performance metric on proposed approach for Codequeries Benchmark.

More details on curated dataset for this benchmark is available on [huggingface](https://huggingface.co/datasets/thepurpleowl/codequeries).

### Steps
-----------
1. Clone the repo in virtual environment.
2. Run `setup.sh` to setup the workspace.
3. Run the following command to get performance metric values.   
**For span prediction**  
`python evaluate_spanprediction.py --example_types_to_evaluate=<positive/negative/all> --setting=<ideal/file_ideal/prefix/sliding_window/twostep>`  
**For relevance prediction**  
`python evaluate_relevance.py`


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