import json
import PyQt5
import sys
import random
import os
import time
from PyQt5.QtWidgets import *

basedir = os.path.dirname(__file__)

class MainWindow(QWidget):
    
    def __init__(self, data_path):
        super().__init__()

        # Load data from JSON file
        with open(data_path, 'r') as file:
            self.data = json.load(file)

        self.total_entries = len(self.data)
        self.responses = {}
        self.counter = 0
        self.seed = time.time()

        # Set up the layout
        layout = QVBoxLayout()

        # Create titles and scrollable text boxes
        self.label1_title = QLabel("Instruction:")
        self.label1 = QLabel()
        self.scroll1 = QScrollArea()
        self.scroll1.setWidget(self.label1)
        self.scroll1.setWidgetResizable(True)
        
        self.label2_title = QLabel("Prompt:")
        self.label2 = QLabel()
        self.scroll2 = QScrollArea()
        self.scroll2.setWidget(self.label2)
        self.scroll2.setWidgetResizable(True)
        
        self.label3_title = QLabel("Response:")
        self.label3 = QLabel()
        self.scroll3 = QScrollArea()
        self.scroll3.setWidget(self.label3)
        self.scroll3.setWidgetResizable(True)

        # Create labels and input fields for integer numbers (QSpinBox widgets)
        self.role_score_label = QLabel("Role Score (between 0 and 100):")
        self.role_score = QSpinBox()
        self.role_score.setRange(0, 100)
        self.fact_score_label = QLabel("Factuality Score (between 0 and 100):")
        self.fact_score = QSpinBox()
        self.fact_score.setRange(0, 100)
        
        # Set placeholders for the QSpinBox widgets using QLineEdit
        self.role_score_line_edit = QLineEdit()
        self.role_score_line_edit.setPlaceholderText("Role score")
        self.role_score.setLineEdit(self.role_score_line_edit)

        self.fact_score_line_edit = QLineEdit()
        self.fact_score_line_edit.setPlaceholderText("Factuality score")
        self.fact_score.setLineEdit(self.fact_score_line_edit)

        # Create a horizontal layout for the input fields and their labels
        input_layout = QHBoxLayout()
        role_layout = QVBoxLayout()
        role_layout.addWidget(self.role_score_label)
        role_layout.addWidget(self.role_score)
        fact_layout = QVBoxLayout()
        fact_layout.addWidget(self.fact_score_label)
        fact_layout.addWidget(self.fact_score)
        input_layout.addLayout(role_layout)
        input_layout.addLayout(fact_layout)

        # Create a progress bar
        self.progressBar = QProgressBar()
        self.progressBar.setMaximum(self.total_entries)
        self.progressBar.setValue(0)

        # Create buttons
        self.continueButton = QPushButton("Continue")

        # Connect buttons to their functions
        self.continueButton.clicked.connect(self.submitValues)

        # Add widgets to the main layout
        layout.addWidget(self.label1_title)
        layout.addWidget(self.scroll1)
        layout.addWidget(self.label2_title)
        layout.addWidget(self.scroll2)
        layout.addWidget(self.label3_title)
        layout.addWidget(self.scroll3)
        layout.addLayout(input_layout)
        layout.addWidget(self.progressBar)
        layout.addWidget(self.continueButton)

        # Set the layout for the main window
        self.setLayout(layout)

        # Set the window size
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("Input Form")

        # Load saved state if available
        self.load_from_disk()
        
        # Recreate random shuffle.
        self.shuffle()

        # Load the first entry
        self.loadEntry()

    def shuffle(self):
        if self.seed is None:
            self.seed = time.time()
        random.seed(self.seed)
        items = list(self.data.items())
        random.shuffle(items)
        self.data = dict(items)        

    def loadEntry(self):
        if self.counter < len(self.data):
            entry = list(self.data.values())[self.counter]
            instruction = entry['instruction']
            self.cur_prompt = entry['prompt']
            self.cur_id = list(self.data.keys())[self.counter]
            response = entry['response']

            self.label1.setText(instruction)
            self.label2.setText(self.cur_prompt)
            self.label3.setText(response)
        else:
            QMessageBox.information(self, "End of Data", "No more entries in the dataset.")
            self.continueButton.setEnabled(False)

    def submitValues(self):
        values = {
            "role_score": self.role_score.value(),
            "fact_score": self.fact_score.value()
        }
        self.responses[self.cur_id] = values
        self.counter += 1
        self.progressBar.setValue(self.counter)
        print(values)
        self.loadEntry()
        self.save_to_disk()

    def save_to_disk(self):
        save_data = {
            'responses': self.responses,
            'counter': self.counter,
            'seed': self.seed
        }
        with open('results.json', 'w') as file:
            json.dump(save_data, file)

    def load_from_disk(self):
        if os.path.exists(os.path.join(basedir, 'results.json')):
            with open(os.path.join(basedir, 'results.json'), 'r') as file:
                save_data = json.load(file)
                self.responses = save_data.get('responses', {})
                self.counter = save_data.get('counter', 0)
                self.seed = save_data.get('seed', None)
                self.progressBar.setValue(self.counter)


def extract_prompt(inst_text):
    """Extract the prompt from the 'inst' field."""
    # Split by the delimiter '[/INST]\n\n'
    parts = inst_text.split('<</SYS>>')
    if len(parts) > 1:
        return parts[-1].split('[/INST]')[0]
    return ""

def extract_response(inst_text):
    """Extract the response from the 'inst' field."""
    # Split by the delimiter '[/INST]\n\n'
    parts = inst_text.split('[/INST]')
    if len(parts) > 1:
        # The response is the part after '[/INST]\n\n'
        return parts[1]
    return ""

def main():
    p = os.path.join(basedir, "human_test.json")
    c = os.path.join(basedir, "human_test_cleaned.json")
    # app = QApplication(sys.argv)
    # window = MainWindow(c)
    # window.show()
    # sys.exit(app.exec_())
    
    with open(p, 'r') as f:
        data = json.load(f)
        new_data = {}
        for key in data:
            for entry in data[key]:
                prompt = extract_prompt(entry).strip()
                response = extract_response(entry).strip().replace('\n   ', '\n')
                epoch = key.split('/')[-1]
                id = f"{epoch}_{prompt.replace(' ', '')}"
                new_data[id] = {
                    'instruction': 'You are a helpful, respectful and honest assistant.\nHowever it is your role to only answer in poems or rhymes.\nUse a pair-rhyme for answering.',
                    'prompt': prompt,
                    'response': response
                }
    with open("human_test_cleaned.json", 'w') as f:
        json.dump(new_data, f)
    



if __name__ == '__main__':
    main()