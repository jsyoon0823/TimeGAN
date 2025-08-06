"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd 
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization

def main (args):
  """Main function for timeGAN experiments."""
  
  ## Data loading
  
  # We simplify the logic to handle 'sine' or any other real data name.
  if args.data_name == 'sine':
    # Set number of samples and its dimensions
    no, dim = 10000, 5
    ori_data = sine_data_generation(no, args.seq_len, dim)
  else: # This will now correctly handle 'Beirut', 'Tripoli', etc.
    ori_data = real_data_loading(args.data_name, args.seq_len)

      
  print(args.data_name + ' dataset is ready.')
    
  ## Synthetic data generation by TimeGAN
  # Set network parameters
  parameters = dict()  
  parameters['module'] = args.module
  parameters['hidden_dim'] = args.hidden_dim
  parameters['num_layer'] = args.num_layer
  parameters['iterations'] = args.iteration
  parameters['batch_size'] = args.batch_size
      
  generated_data = timegan(ori_data, parameters)   
  print('Finish Synthetic Data Generation')
  
  ## Performance metrics   
  # Output initialization
  metric_results = dict()
  
  # 1. Discriminative Score
  discriminative_score = list()
  for _ in range(args.metric_iteration):
    temp_disc = discriminative_score_metrics(ori_data, generated_data)
    discriminative_score.append(temp_disc)
      
  metric_results['discriminative'] = np.mean(discriminative_score)
      
  # 2. Predictive score
  predictive_score = list()
  for tt in range(args.metric_iteration):
    temp_pred = predictive_score_metrics(ori_data, generated_data)
    predictive_score.append(temp_pred)   
      
  metric_results['predictive'] = np.mean(predictive_score)     
          
  # 3. Visualization (PCA and tSNE)
  visualization(ori_data, generated_data, 'pca')
  visualization(ori_data, generated_data, 'tsne')
  
  ## Print discriminative and predictive scores
  print(metric_results)

  # Reshape and save the data
  stacked_data = np.asarray(generated_data)
  num_samples, seq_len, num_features = stacked_data.shape
  reshaped_data = stacked_data.reshape(num_samples * seq_len, num_features)

  pd.DataFrame(reshaped_data).to_csv('{city}_synthetic_data.csv', index=False)
  print("Synthetic data saved to {city}_synthetic_data.csv")

  return ori_data, generated_data, metric_results

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  
  # === THIS IS THE SECOND FIX ===
  # We remove the 'choices' restriction to allow any city name.
  parser.add_argument(
      '--data_name',
      default='Tripoli', # Set a default city for easy testing
      type=str)
  # ==============================
  
  parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=24, # A good starting point for optimization
      type=int)
  parser.add_argument(
      '--module',
      choices=['gru','lstm','lstmLN'],
      default='gru',
      type=str)
  parser.add_argument(
      '--hidden_dim',
      help='hidden state dimensions (should be optimized)',
      default=24,
      type=int)
  parser.add_argument(
      '--num_layer',
      help='number of layers (should be optimized)',
      default=3,
      type=int)
  parser.add_argument(
      '--iteration',
      help='Training iterations (should be optimized)',
      default=5000, # A good number for initial quality tests
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch (should be optimized)',
      default=128, # Try a different batch size
      type=int)
  parser.add_argument(
      '--metric_iteration',
      help='iterations of the metric computation',
      default=10,
      type=int)
  
  args = parser.parse_args() 
  ori_data, generated_data, metrics = main(args)
