# Deck Auto Naming
## Notice

- This is UNOFFICIAL project and you use at your own risk.
- This software has been tested only with `python==3.7.4`.

## Install
Download `got/` & `got3/` from [GetOldTweets-python](https://github.com/Jefferson-Henrique/GetOldTweets-python) and place them into `modules/`.

Then,

```bash
$ pip install -r requirements.txt
```

## Usage
Run `python main.py -h` to see more options.

### Find decks from recent tweets
```bash
$ python main.py find
```

### Save decks for the specified code
```bash
$ python main.py save --code "aaDcc4-EitrO6-xGKxKc"
```

### Show auto-generated deck namings
```bash
$ python main.py name
```

or

```bash
$ python main.py name --code "aaDcc4-EitrO6-xGKxKc"
```

## License
MIT
