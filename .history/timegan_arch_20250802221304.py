from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def visualize_timegan_architecture():
    # Create a directed graph
    dot = Digraph(comment='TimeGAN Architecture', format='png')
    dot.attr(rankdir='TB', size='12,12')
    
    # Global attributes
    dot.attr('node', shape='box', style='filled', color='lightgrey')
    
    # Define components
    with dot.subgraph(name='cluster_real_data') as c:
        c.attr(label='Real Time-series Data', color='blue')
        c.node('X', 'Input Data\n(batch_size, seq_len, feature_dim)')
        c.node('X_emb', 'Embedded Data\n(batch_size, seq_len, hidden_dim)')
        c.attr(color='blue')
    
    with dot.subgraph(name='cluster_random_noise') as c:
        c.attr(label='Random Noise', color='green')
        c.node('Z', 'Random Noise\n(batch_size, seq_len, latent_dim)')
        c.attr(color='green')
    
    with dot.subgraph(name='cluster_embedder') as c:
        c.attr(label='Embedder (Autoencoder)', color='orange')
        c.node('E', 'Embedder\n(LSTM/GRU based)')
        c.node('H', 'Hidden States\n(batch_size, seq_len, hidden_dim)')
        c.attr(color='orange')
    
    with dot.subgraph(name='cluster_recovery') as c:
        c.attr(label='Recovery', color='orange')
        c.node('R', 'Recovery\n(LSTM/GRU based)')
        c.node('X_tilde', 'Recovered Data\n(batch_size, seq_len, feature_dim)')
        c.attr(color='orange')
    
    with dot.subgraph(name='cluster_generator') as c:
        c.attr(label='Generator', color='red')
        c.node('G', 'Generator\n(LSTM/GRU based)')
        c.node('X_hat', 'Generated Data\n(batch_size, seq_len, feature_dim)')
        c.node('H_hat', 'Generated Hidden States\n(batch_size, seq_len, hidden_dim)')
        c.attr(color='red')
    
    with dot.subgraph(name='cluster_supervisor') as c:
        c.attr(label='Supervisor', color='purple')
        c.node('S', 'Supervisor\n(LSTM/GRU based)')
        c.node('H_super', 'Supervised Hidden States\n(batch_size, seq_len, hidden_dim)')
        c.attr(color='purple')
    
    with dot.subgraph(name='cluster_discriminator') as c:
        c.attr(label='Discriminator', color='darkgreen')
        c.node('D', 'Discriminator\n(LSTM/GRU based)')
        c.node('Y_real', 'Real/Fake Prediction\nfor real data')
        c.node('Y_fake', 'Real/Fake Prediction\nfor generated data')
        c.attr(color='darkgreen')
    
    # Define connections
    # Embedder and Recovery
    dot.edge('X', 'E', label='Input')
    dot.edge('E', 'H', label='Encodes to')
    dot.edge('H', 'R', label='Input')
    dot.edge('R', 'X_tilde', label='Decodes to')
    
    # Generator
    dot.edge('Z', 'G', label='Input')
    dot.edge('G', 'H_hat', label='Generates')
    dot.edge('H_hat', 'S', label='Input')
    dot.edge('S', 'H_super', label='Predicts next step')
    
    # Discriminator
    dot.edge('H', 'D', label='Input (real)', style='dashed')
    dot.edge('H_super', 'D', label='Input (fake)', style='dashed')
    dot.edge('D', 'Y_real', label='Output')
    dot.edge('D', 'Y_fake', label='Output')
    
    # Additional connections
    dot.edge('H', 'X_emb', label='Also used as')
    
    # Loss functions (simplified)
    with dot.subgraph(name='cluster_losses') as c:
        c.attr(label='Loss Functions', color='brown')
        c.node('L_auto', 'Autoencoder Loss\n(MSE X vs X_tilde)')
        c.node('L_adv', 'Adversarial Loss\n(Cross-entropy Y_real vs Y_fake)')
        c.node('L_super', 'Supervisor Loss\n(MSE H vs H_super)')
        c.node('L_emb', 'Embedding Loss\n(Combination of above)')
        c.attr(color='brown')
    
    dot.edge('X_tilde', 'L_auto')
    dot.edge('X', 'L_auto')
    dot.edge('Y_real', 'L_adv')
    dot.edge('Y_fake', 'L_adv')
    dot.edge('H', 'L_super')
    dot.edge('H_super', 'L_super')
    dot.edge('L_auto', 'L_emb')
    dot.edge('L_adv', 'L_emb')
    dot.edge('L_super', 'L_emb')
    
    # Render the graph
    dot.render('timegan_architecture', view=False, cleanup=True)
    
    # Display in matplotlib (for VSCode)
    img = mpimg.imread('timegan_architecture.png')
    plt.figure(figsize=(15, 15))
    plt.imshow(img)
    plt.axis('off')
    plt.title('TimeGAN Architecture')
    plt.show()

if __name__ == '__main__':
    # Check if graphviz is installed
    try:
        visualize_timegan_architecture()
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease install the required packages:")
        print("pip install graphviz matplotlib")
        print("Also make sure Graphviz is installed on your system:")
        print("https://graphviz.org/download/")