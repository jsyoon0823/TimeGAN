from graphviz import Digraph

# Create a directed graph
dot = Digraph(comment='TimeGAN Architecture', format='pdf')

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

# Real data path edges
dot.edges([('X', 'E'), ('E', 'H'), ('H', 'R'), ('R', 'XT')])

# Synthetic data path edges
dot.edges([('Z', 'G'), ('G', 'HT'), ('HT', 'S'), ('S', 'HH')])

# Discriminator edges
dot.edge('H', 'D')
dot.edge('HH', 'D')

# Render as PDF
dot.render("timegan_architecture.pdf", view=True)

