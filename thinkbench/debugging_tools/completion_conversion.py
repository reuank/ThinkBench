import json
import math
from typing import List, Dict

completion_probabilities = [
  {
    "content": " Sure",
    "probs": [
      {
        "tok_str": " Sure",
        "prob": 0.9033634066581726
      },
      {
        "tok_str": " The",
        "prob": 0.07014923542737961
      },
      {
        "tok_str": " To",
        "prob": 0.005697799380868673
      }
    ]
  },
  {
    "content": ",",
    "probs": [
      {
        "tok_str": ",",
        "prob": 0.6936726570129395
      },
      {
        "tok_str": "!",
        "prob": 0.3062383532524109
      },
      {
        "tok_str": " let",
        "prob": 5.714746657758951e-05
      }
    ]
  }
]

top_logprobs: List[Dict[str, float]] = []
for completion_probability in completion_probabilities:
    top_logprobs.append(
        {item["tok_str"]:math.log(item["prob"]) for item in completion_probability["probs"]}
    )

print(json.dumps(top_logprobs, indent=2))
