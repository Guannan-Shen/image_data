# Working Environment
## [Follow the instruction for Virtualenv](https://www.tensorflow.org/install/install_linux#InstallingVirtualenv)
1. Using the **virtualenv** as recommended.
2. Activate and install msgpack, tf
    
        $ virtualenv --system-site-packages -p python3
        $ source ~/tensorflow/venv/bin/activate      # bash, sh, ksh, or zsh, tensorflow under home folder
        (venv)$ pip install -U pip
        (venv)$ pip install msgpack
        (venv)$ deactivate  # stop the virtualenv

