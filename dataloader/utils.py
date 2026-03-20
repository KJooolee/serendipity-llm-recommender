import json
import os.path as osp
from typing import Union
import pickle

def load_split_retrieved(val_probs_path,val_labels_path,val_metrics_path, test_probs_path,test_labels_path,test_metrics_path):
    import pickle

    with open(val_probs_path, 'rb') as f:
        val_probs_data = pickle.load(f)
    with open(val_labels_path, 'rb') as f:
        val_labels_data = pickle.load(f)
    with open(val_metrics_path, 'rb') as f:
        val_metrics_data = pickle.load(f)
    with open(test_probs_path, 'rb') as f:
        test_probs_data = pickle.load(f)
    with open(test_labels_path, 'rb') as f:
        test_labels_data = pickle.load(f)                
    with open(test_metrics_path, 'rb') as f:
        test_metrics_data = pickle.load(f)
    merged_data = {
        'val_probs': val_probs_data.get('val_probs', []),
        'val_labels': val_labels_data.get('val_labels', []),
        'val_metrics': val_metrics_data.get('val_metrics', {}),
        'test_probs': test_probs_data.get('test_probs', []),
        'test_labels': test_labels_data.get('test_labels', []),
        'test_metrics': test_metrics_data.get('test_metrics', {})
    }

    return merged_data
class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # template_name = "alpaca"
            template_name = "alpaca_short"
        file_name = osp.join("dataloader", "templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )
            
    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        if input and label:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input, label=label
            )
        elif input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input, label=""
            )
        elif label:
            res = self.template["prompt_no_input"].format(
                instruction=instruction, label=label
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction, label=""
            )

        if self._verbose:
            print(res)
        return res

    # def generate_prompt(
    #     self,
    #     instruction: str,
    #     input: Union[None, str] = None,
    #     label: Union[None, str] = None,
    # ) -> str:
    #     if input:
    #         res = self.template["prompt_input"].format(
    #             instruction=instruction, input=input
    #         )
    #     else:
    #         res = self.template["prompt_no_input"].format(
    #             instruction=instruction
    #         )
    #     if label:
    #         res = f"{res}{label}"
    #     if self._verbose:
    #         print(res)
    #     return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()