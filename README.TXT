In order to create the embeddings run:

top1_embeddings.py

Once the embeddings have been created you can see how well the code that computes retreival accuracy of
dot product, cosine similarity, and euclidean distance with:

ann.py

which runs main_mips.py with passed in parameters.
it will generate clusters for the first test, which is for dot product, but shouldn't need to recalculate them after. 
you may need to create sub folders 'ells_stats_sig' and 'results'.

If you are interested in trying out the not very good attempt at using query aware training you can run:

query_ann.py

it runs new_mips.py with passed in params.

Thank you for everything, hope you enjoy the break!
