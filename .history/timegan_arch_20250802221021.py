from graphviz import Digraph

# Define the TimeGAN architecture diagram using Graphviz
dot = Digraph(comment='TimeGAN Architecture', format='png')

# Nodes for real data path
dot.node('X', 'Real Time-Series Data X')
dot.node('E', 'Embedder')
dot.node('H', 'Latent Representation H')
dot.node('R', 'Recovery')
dot.node('XT', 'Reconstructed X_tilde')

# Nodes for synthetic data path
dot.node('Z', 'Random Noise Z')
dot.node('G', 'Generator')
dot.node('HT', 'Latent Synthetic H_tilde')
dot.node('S', 'Supervisor')
dot.node('HH', 'H_hat (time-consistent latent)')

# Discriminator
dot.node('D', 'Discriminator')

# Edges for real data path
dot.edges([('X', 'E'), ('E', 'H'), ('H', 'R'), ('R', 'XT')])

# Edges for synthetic data path
dot.edges([('Z', 'G'), ('G', 'HT'), ('HT', 'S'), ('S', 'HH')])

# Edges to Discriminator
dot.edge('H', 'D')
dot.edge('HH', 'D')

# Render the diagram as PNG
dot.render("timegan_architecture.png", view=True)
