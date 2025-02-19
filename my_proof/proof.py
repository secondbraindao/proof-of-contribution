import logging
import os
from typing import Dict, Any

from my_proof.proof_of_ownership import verify_ownership
from my_proof.proof_of_quality_n_authenticity import process_files_for_quality_n_authenticity_scores
from my_proof.models.proof_response import ProofResponse
from my_proof.proof_of_uniqueness import process_files_for_uniqueness

# Ensure logging is configured
logging.basicConfig(level=logging.INFO)

CONTRIBUTION_THRESHOLD = 4
EXTRA_POINTS = 5

class Proof:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.proof_response = ProofResponse(dlp_id=config['dlp_id'])
        self.wallet_address = ""
    
    def read_author_from_file(self, file_path: str):
        """
        Read parameters from a text file.

        :param file_path: Path to the text file
        :return: Tuple containing author, signature, and random_string
        """
        params = {}
        with open(file_path, "r") as file:
            for line in file:
                key, value = line.strip().split(": ", 1)
                params[key] = value
        return params["author"]

    def generate(self) -> ProofResponse:
        """Generate proofs for all input files."""
        logging.info("Starting proof generation")

        proof_response_object = {
            'dlp_id': self.config.get('dlp_id', 30),
            'valid': True,
        }

        # Read the wallet address from the first .txt file in the input directory
        txt_files = [f for f in os.listdir(self.config['input_dir']) if f.endswith('.txt')]
        if txt_files:
            self.wallet_address = self.read_author_from_file(os.path.join(self.config['input_dir'], txt_files[0])).lower()
            logging.info(f"Wallet Address {self.wallet_address}")
            
            
        for input_filename in os.listdir(self.config['input_dir']):
            logging.info(f"Processing file: {input_filename}")
            file_id = self.config.get('file_id') 
            logging.info(f"Processing file ID: {file_id}")
            uniqueness_details = process_files_for_uniqueness(file_id, self.config['input_dir'], self.wallet_address)
            quality_n_authenticity_details = process_files_for_quality_n_authenticity_scores(uniqueness_details.get("unique_csv_data"), uniqueness_details.get("unique_json_data"), uniqueness_details.get("unique_yaml_data"))

            # proof_response_object['ownership'] = 1.0
            proof_response_object['ownership'] = verify_ownership(self.config['input_dir'])
            proof_response_object['uniqueness'] = uniqueness_details.get("uniqueness_score")
            proof_response_object['quality'] = quality_n_authenticity_details.get("quality_score")
            proof_response_object['authenticity'] = quality_n_authenticity_details.get("authenticity_score")

            if proof_response_object['authenticity'] < 1.0:
              proof_response_object['valid'] = True

                # Calculate the final score
            proof_response_object['score'] = self.calculate_final_score(proof_response_object)

            # proof_response_object['attributes'] = {
            #    # 'normalizedContributionScore': contribution_score_result['normalized_dynamic_score'],
            #    # 'totalContributionScore': contribution_score_result['total_dynamic_score'],
            # }

        logging.info(f"Proof response: {proof_response_object}")
        return proof_response_object
        
    def calculate_final_score(self, proof_response_object: Dict[str, Any]) -> float:
        attributes = ['authenticity', 'uniqueness', 'quality', 'ownership']
        weights = {
            'authenticity': 0.003,  # Low weight for authenticity
            'ownership': 0.005,  # Slightly higher than authenticity
            'uniqueness': 0.342,  # Moderate weight for uniqueness
            'quality': 0.650  # High weight for quality
        }

        weighted_sum = 0.0
        for attr in attributes:
            weighted_sum += proof_response_object.get(attr, 0) * weights[attr]

        return weighted_sum
