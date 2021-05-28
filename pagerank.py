import os
import random
import re
import sys
from pomegranate import *
from collections import Counter

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    result = {}

    # Looping through each page in corpus to assign probablity
    for name in corpus:
        # Default probablity
        value = (1 - damping_factor) / len(corpus)
        value = round(value, 2)

        # If page contains link to other pages update their probablity
        if name in corpus[page]:

            # Probablity updated
            value += damping_factor / (len(corpus[page]))
            # Probablity added
            result[name] = value
        
        # else put defalut value in their probablity
        else:
            # Prabablity added
            result[name] = value

    return result

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Defining starting probabilities
    starting = {}
    for page in corpus:
        value = 1 / len(corpus)
        starting[page] = value

    start = DiscreteDistribution(starting)

    # Defining transition model
    transtition = []
    for page1 in corpus:
        model = transition_model(corpus, page1, damping_factor)
        for page2 in corpus:
            tran = []
            tran.append(page1)
            tran.append(page2)
            tran.append(model[page2])
            transtition.append(tran)

    transitions = ConditionalProbabilityTable(transtition, [start])

    # Creating Markov chain
    model = MarkovChain([start, transitions])

    # Sampling n states from chain
    samples = model.sample(n)

    # Getting count of each page called
    counts = Counter(samples)

    # Making ranks dictionary
    ranks = {}
    for page in corpus:
        ranks[page] = counts[page] / n
    return ranks



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initializing
    ranks = {}
    corpus_incoming = {}
    corpus_outgoing = corpus

    # Assigning Starting PageRanks
    for page in corpus:
        ranks[page] = 1 / len(corpus)
        corpus_incoming[page] = set()

    for page in corpus:

        # If page contains no links
        if len(corpus[page]) == 0:
            for p in corpus:
                # Should point to each page including own idea implemented
                corpus_incoming[p].add(page)
                corpus_outgoing[page].add(p)
        else:
            for link in corpus[page]:
                # Making a new corpus that saves the page with the set of pages that contain a link to it
                corpus_incoming[link].add(page)
    
    # Looping to increase accuracy
    for _ in range(100):

        # Looping through each page to get PageRank
        for page in corpus_incoming:

            # Getting the sum of PageRanks of incoming links
            sum = 0
            for link in corpus_incoming[page]:
                value = ranks[link] / len(corpus_outgoing[link])
                sum += value

            # PageRank iterative formula
            rank = ((1 - damping_factor) / len(corpus_incoming)) + (damping_factor * sum)

            # Updating Ranks
            ranks[page] = rank

    return ranks

if __name__ == "__main__":
    main()
