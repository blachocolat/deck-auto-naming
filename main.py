# -*- coding: utf-8 -*-
import collections
import datetime
import glob
import json
import numpy as np
import re
import requests
import ssl
import sys
import time

from argparse import ArgumentParser
from bs4 import BeautifulSoup
from enum import Enum
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm

if sys.version_info[0] < 3:
  from modules import got
else:
  from modules import got3 as got

# WARN: Ignore `SSL: CERTIFICATE_VERIFY_FAILED` error
ssl._create_default_https_context = ssl._create_unverified_context

class Category(Enum):
  POKEMON = 0
  GOODS = 1
  SUPPORT = 2
  STADIUM = 3
  ENERGY = 4

  def key(self):
    return ['pokemons', 'goods', 'supports', 'stadiums', 'energies'][int(self.value)]

class Deck():
  def __init__(self, dict_):
    self.__dict = dict_

  @property
  def code(self):
    return self.__dict['code']

  @staticmethod
  def from_json(filename):
    with open(filename, mode='r') as f:
      deck = Deck(json.loads(f.read(), encoding='utf-8'))
    
    return deck

  @staticmethod
  def fetch(code, timestamp=None):
    r = requests.get(f"https://www.pokemon-card.com/deck/print.html/deckID/{code}/")
    soup = BeautifulSoup(r.text, 'html.parser')

    dict_ = collections.OrderedDict()
    dict_['code'] = code

    selector = 'body > div.WrapperArea > div > div > section:nth-child(2) > div > div > table > tbody'
    tables = soup.select(selector)

    for i, category in enumerate(Category):
      cards = []

      for row in tables[i].select('tr'):
        card = Deck.__parse_row(row, with_expansion=(category == Category.POKEMON))

        if not card is None:
          cards.append(card)

      dict_[category.key()] = cards

    if not timestamp is None:
      dict_['timestamp'] = timestamp
    
    return Deck(dict_)

  @staticmethod
  def __parse_row(row, with_expansion=False):
    name = row.select_one('td:first-child')
    name = name.text.strip('\xa0') if not name is None else ''

    count = row.select_one('td:last-child')
    count = count.text.strip('\xa0') if not count is None else ''

    if len(name) > 0 and len(count) > 0:
      card = collections.OrderedDict()
      card['name'] = name

      if with_expansion:
        expansion = row.select_one('td:nth-child(2)')
        card['expansion'] = expansion.text.strip('\xa0')

      card['count'] = int(count)
      return card
    else:
      return None

  def save(self, to, filename=None):
    filename = filename if not filename is None else f"{self.code}.json"

    with open(f"{to}/{filename}", mode='w', encoding='utf-8') as f:
      json.dump(self.__dict, f, indent=4, ensure_ascii=False)

  def to_tsv(self, ignore_expansion=True):
    regex = r'^基本(草|炎|水|雷|超|闘|悪|鋼|フェアリー)エネルギー$'
    pattern = re.compile(regex)
    line = ''

    for category in Category:
      for card in self.__dict[category.key()]:
        for _ in range(card['count']):
          if pattern.match(card['name']):
            continue
          
          prefix = category.key()[:2]

          if not ignore_expansion and 'expansion' in card:
            line += f"{prefix}{card['name']}<{card['expansion']}>\t"
          else:
            line += f"{prefix}{card['name']}\t"
    
    return line

class AutoNaming():
  def __init__(self, path_to_dataset='./decks/*.json'):
    filenames = glob.glob(path_to_dataset)
    self.__decks = []

    for filename in filenames:
      self.__decks.append(Deck.from_json(filename))

    dataset = [deck.to_tsv() for deck in self.__decks]
    dataset = np.array(dataset).ravel()

    vectorizer = CountVectorizer(lowercase=False, token_pattern=u'(?u)[^\\t]+')
    self.__tf_X = vectorizer.fit_transform(dataset).toarray()

    transformer = TfidfTransformer()
    self.__tfidf_X = transformer.fit_transform(self.__tf_X).toarray()

    sort_X = self.__tfidf_X.argsort(axis=1)[:,::-1]
    self.__tf_X = np.array([x[a] for x, a in zip(self.__tf_X, sort_X)])
    self.__tfidf_X = np.array([x[a] for x, a in zip(self.__tfidf_X, sort_X)])
    self.__feature_cards = np.array(vectorizer.get_feature_names())[sort_X]

  def display(self, codes=None, n_top_cards=10, n_samples=10, max_picks=20):
    if not codes is None:
      indexes = [i for i, deck in enumerate(self.__decks) if (deck.code in codes)]
    else:
      indexes = [np.random.randint(len(self.__decks)) for _ in range(n_samples)]

    for i in indexes:
      print(self.__decks[i].code)
      
      fcards = self.__feature_cards[i,:n_top_cards]
      rep_pokemons = []
      rep_others = []
      n_picks = 0

      for j, fcard in enumerate(fcards):
        tf = self.__tf_X[i,j]

        if tf == 0:
          continue

        tfidf = self.__tfidf_X[i,j]
        prefix = fcard[:2]
        name = fcard[2:]

        if n_picks < max_picks:
          n_picks += tf

          if prefix == Category.POKEMON.key()[:2]:
            if len(rep_pokemons) < 6:
              rep_pokemons.append(name)
          else:
            if len(rep_others) < 1:
              rep_others.append(name)

        if tf > 0:
          represented = name in rep_pokemons or name in rep_others
          print(f"[{tfidf:.4f}] {'*' if represented else ' '}{name} ({tf})")

      print(u'')

def do_find(args):
  since = args.since if args.since is not None else (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
  tweet_criteria = got.manager.TweetCriteria().setQuerySearch('www.pokemon-card.com/deck/confirm.html/deckID/').setSince(since).setMaxTweets(args.max_tweets)
  tweets = got.manager.TweetManager.getTweets(tweet_criteria)

  print(f"{len(tweets):d} tweet(s) be found...")

  regex = r'www.pokemon-card.com/deck/confirm.html/deckID/([a-zA-Z0-9]{6}-[a-zA-Z0-9]{6}-[a-zA-Z0-9]{6})/'
  pattern = re.compile(regex)
  n_fails = 0

  for tweet in tqdm(tweets):
    matchObj = pattern.search(tweet.text.replace(' ', ''))
    code = matchObj and matchObj.group(1)

    if code is None:
      n_fails += 1
      continue
    
    deck = Deck.fetch(code, timestamp=int(tweet.date.strftime('%s%f')[:-3]))
    deck.save(to=args.dir)

    time.sleep((np.random.rand() - 0.5) * 2.0 * args.ave_sleep + args.ave_sleep)

  print(f"{len(tweets) - n_fails:d} deck(s) be saved to `{args.dir}`.")

def do_save(args):
  for code in tqdm(args.code):
    deck = Deck.fetch(code)
    deck.save(to=args.dir)

    time.sleep((np.random.rand() - 0.5) * 2.0 * args.ave_sleep + args.ave_sleep)

  print(f"{len(args.code):d} deck(s) be saved to `{args.dir}`.")

def do_name(args):
  naming = AutoNaming()

  if not args.code is None:
    naming.display(codes=args.code, n_top_cards=args.n_top_cards, n_samples=args.n_samples, max_picks=args.max_picks)
  else:
    naming.display(n_top_cards=args.n_top_cards, n_samples=args.n_samples, max_picks=args.max_picks)

if __name__ == '__main__':
  parser = ArgumentParser(description='')
  subparsers = parser.add_subparsers()

  parser_save = subparsers.add_parser('find', help='find decks from recent tweets and save them (see `find -h`)')
  parser_save.add_argument('-d', '--dir', default='decks/', help='path to the saving directory')
  parser_save.add_argument('--since', default=None)
  parser_save.add_argument('--max-tweets', type=int, default=10)
  parser_save.add_argument('--ave-sleep', type=float, default=1.0)
  parser_save.set_defaults(handler=do_find)

  parser_save = subparsers.add_parser('save', help='save decks specified by the deck codes (see `save -h`)')
  parser_save.add_argument('-c', '--code', required=True, nargs='+', help='deck code(s) you want to save')
  parser_save.add_argument('-d', '--dir', default='decks/', help='path to the saving directory')
  parser_save.add_argument('--ave-sleep', type=float, default=1.0)
  parser_save.set_defaults(handler=do_save)

  parser_name = subparsers.add_parser('name', help='show auto-generated deck namings (see `name -h`)')
  parser_name.add_argument('-c', '--code', nargs='+', help="deck code(s) you want to name")
  parser_name.add_argument('-d', '--dir', default='decks/', help='path to the saving directory')
  parser_name.add_argument('--n-top-cards', type=int, default=10)
  parser_name.add_argument('--n-samples', type=int, default=10)
  parser_name.add_argument('--max-picks', type=int, default=20)
  parser_name.set_defaults(handler=do_name)

  args = parser.parse_args()

  if hasattr(args, 'handler'):
    args.handler(args)
  else:
    parser.print_help()
