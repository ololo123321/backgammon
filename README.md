TD-learning of backgammon. tf implementation.
http://www.bkgm.com/articles/tesauro/tdl.html

train:
```bash
bash train.sh
```

play:
```bash
bash play.sh
```

TODO:
- [x] tests of State
- [x] cythonized state.py
- [] evaluate agent during training by playing with previous checkpoint
- [x] exporting inference graph
- [] docker support (cpu and gpu)
- [] fix mcts agent