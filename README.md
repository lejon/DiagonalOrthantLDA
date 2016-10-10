[![Build Status](https://travis-ci.org/lejon/DiagonalOrthantLDA.svg?branch=master)](https://travis-ci.org/lejon/DiagonalOrthantLDA)

# DOLDA

Supervised LDA using DO-Probit

This is the repo for our Diagonal Orthant Latent Dirichlet Allocation (DOLDA) implementation described in the article [arXiv:1602.00260](https://arxiv.org/abs/1602.00260 "arXiv:1602.00260"): 

```
   @unpublished{MagnussonJonsson2016,
      author = "Måns Magnusson, Leif Jonsson, Mattias Villani",
      title = "DOLDA - a regularized supervised topic model for high-dimensional multi-class regression",
      note = "http://github.com/lejon/DOLDA",
      year = 2016}
```
The toolkit is Open Source Software, and is released under the Common Public License. You are welcome to use the code under the terms of the license for research or commercial purposes, however please acknowledge its use with a citation:
  Magnusson, Jonsson, Villani.  "DOLDA - a regularized supervised topic model for high-dimensional multi-class regression"
  
For very large datasets you might need to add the -Xmx60g flag to Java

Please remember that this is a research prototype and the standard disclaimers apply.
You will see printouts during unit tests, commented out code, old stuff not cleaned out yet etc.

## Repo extras

- extralibs
  Contains two external dependencies which are not available from Maven Central or any other public repo yet.
  
## Installation

1. Install Apache Maven and run:

```mvn package```

in a shell of your choice.
