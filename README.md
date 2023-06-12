# CodeQueries Benchmark

CodeQueries is a dataset to evaluate the ability of neural networks to answer semantic queries over code. Given a query and code, a model is expected to identify answer and supporting-fact spans in the code for the query. This is extractive question-answering over code, for questions with a large scope (entire files) and complexity including both single- and multi-hop reasoning. More details on the curated dataset for this benchmark are available on [huggingface](https://huggingface.co/datasets/thepurpleowl/codequeries).

<p align="center">
    <img src="codequeries_instance.png"/>
</p>

The repo provides scripts to reproduce the results of NeurIPS dataset track [submission]().

### Steps
-----------
1. Clone the repo in a virtual environment.
2. Run `setup.sh` to setup the workspace.
3. Run the following commands to get performance metric values.   

#### To run Two-step setup evaluation
`python3 evaluate_spanprediction.py --example_types_to_evaluate=<positive/negative> --setting=twostep --span_type=<both/answer/sf> --span_model_checkpoint_path=<model-ckpt-with-low-data/Cubert-1K-low-data/finetuned_ckpts/Cubert-1K> --relevance_model_checkpoint_path=<model-ckpt-with-low-data/Twostep_Relevance-512-low-data/finetuned_ckpts/Twostep_Relevance-512>`


#### To run LLM experiment Evaluation
`python evaluate_generated_spans.py --g=test_dir_file_0shot/logs`  
`python evaluate_generated_spans.py --g=test_dir_file_fewshot_sf/logs --with_sf=True`

### Experiment results on sampled data
-----------
#### LLM experiment
<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="2">Zero-shot prompting</th>
      <th colspan="2">Few-shot prompting with BM25 retrieval</th>
      <th> Few-shot prompting with supporting fact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Pass@k</td>
      <td>Positive</td>
      <td>Negative</td>
      <td>Positive</td>
      <td>Negative</td>
      <td>Positive</td>
    </tr>
    <tr>
      <td>1</td>
      <td>9.82</td>
      <td>12.83</td>
      <td>16.45</td>
      <td>44.25</td>
      <td>21.88</td>
    </tr>
    <tr>
      <td>2</td>
      <td>13.06</td>
      <td>17.42</td>
      <td>21.14</td>
      <td>55.53</td>
      <td>28.06</td>
    </tr>
    <tr>
      <td>5</td>
      <td>17.47</td>
      <td>22.85</td>
      <td>27.69</td>
      <td>65.43</td>
      <td>34.94</td>
    </tr>
    <tr>
      <td>10</td>
      <td>20.84</td>
      <td>26.77</td>
      <td>32.66</td>
      <td>70.0</td>
      <td>39.08</td>
    </tr>
  </tbody>
</table>

 #### Two-step setup
 <table>
  <thead>
    <tr>
      <th></th>
      <th colspan="2">Answer span prediction</th>
      <th>Answer & supporting-fact span prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Variant</td>
      <td>Positive</td>
      <td>Negative</td>
      <td>Positive</td>
    </tr>
    <tr>
      <td>Two-step(20, 20)</td>
      <td>9.42</td>
      <td>92.13</td>
      <td>8.42</td>
    </tr>
    <tr>
      <td>Two-step(all, 20)</td>
      <td>15.03 </td>
      <td>94.49</td>
      <td>13.27</td>
    </tr>
    <tr>
      <td>Two-step(20, all)</td>
      <td>32.87</td>
      <td>96.26</td>
      <td>30.66</td>
    </tr>
    <tr>
      <td>Two-step(all, all)</td>
      <td>51.90</td>
      <td>95.67</td>
      <td>49.30</td>
    </tr>
  </tbody>
</table>

