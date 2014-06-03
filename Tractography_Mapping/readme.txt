bundle_registration_elef.py: only run in dipy/bundle_registration branch (git fetch elef)  from eleftherios repository

Hi Sandro and Bao,
(Paolo in Cc)

Here you can find the current repository of the code I am writing to play
with the concept of tractography mapping:
   git clone <login>@nilab.cimec.unitn.it:/srv/project/paper/2014/tractography_mapping

The repository has a subdirectory code/ with the code inside. The file I showed you
yesterday is "tractography_mapping.py", which perform simulated annealing optimization
of the loss function of tractography mapping. Notice that in order to run the code
you need simluated_annealing.py that you can find here:
   https://github.com/emanuele/simulated_annealing
So just copy that file in the code/ directory. I don't (want to) put that file in
the repository on purpose.

By the way, you need to install the python-dipy package to run the code. Use the
neurodebian repository.

Notice that a toy dataset is available under the directory code/data/. I guess it
is enough to test many different ideas.

Within the code/ directory there are two more files:

- streamline_matching.py : which attempt to compute and show the similarity
between streamlines based on the concept of graph matching applied to the
respective neighborhoods. Note that the code is ugly, so wait for a while before
digging into it. I plan to clean it up soon.

- simple_mapping.py : this is another sub-optimal attempt of mapping that
I am investigating. Just skip it for now.

I am also starting a report, which you can find in paper/ . But it's almost empty
now.

Let me know about your ideas and attempts.

Best,

Emanuele
