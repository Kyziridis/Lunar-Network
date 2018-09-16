	########## Lunar-Network-Lander-Network ###########
## ReadME

1. In order to run both scripts you need to install openAI-gym lib.

2. For gym install follow the next steps:

    $ pip install --upgrade pip
    $ git clone https://github.com/openai/gym.git
    $ cd gym
    $ pip install -e '.[box2d]'

    # test that it worked 
    $ python >>> import gym >>> gym.make('LunarLander-v2')

    # The above instructions are for linux systems and MacOS. 
    # We do not use proprietary software so we do not have instructions for windows.  

3. Other dependencies : Keras, matplotlib, numpy, random, rensor flow

4. Do not use any Python-IDE to run the code if you want to see the game-simulation (environment-render). Run the code in a console or a terminal by typping: $ python script-name.py

################################Comments: 
The DQN.py uses either a ready model for training or it builds a new one. Under /pretrain folder you can find 5 different models which won the game. If you want to use one pretrain model of those, copy it to the parent directory where the DQN.py is located and rename it as "lunar-lander_solved.h5". If you use a pretrained model, when you run the script you will see the message >>> LOAD an existing model, and under this you will see the summary of that model. If you do not include a pretrained model the script will build a new model and the corresponding message will be >>> Build a new model.

The Naiv-NN.py it does not use any pretrained models. Under the folder "pretrain" you can find also some plots and some of the videos we monitored.



#### SUPPORT >_ GNU/Linux >_ Free Software Foundation ####

