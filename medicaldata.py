import json
import os

class MedicalDataset:

    def __init__(self, dataname):
        self.dataset_dir = "data"
        self.path = os.path.join(self.dataset_dir, dataname)
        self.cases_list = None
        self.load_data()

    def load_data(self):
        with open(self.path, 'r') as json_file:
            data = json.load(json_file)
            self.cases_list = data["Cases"]

    def __len__(self):
        return len(self.cases_list)
        
    def __getitem__(self, ndx):
        if isinstance(ndx, slice):
            start, stop, step = ndx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        
        case = self.cases_list[ndx]
        patient_presentation = case["Initial Presentation"]
        ground_truth_diagnosis = case["Final Name"]
        
        return patient_presentation, ground_truth_diagnosis
        
