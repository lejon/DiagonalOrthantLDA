[![Build Status](https://travis-ci.org/lejon/DiagonalOrthantLDA.svg?branch=master)](https://travis-ci.org/lejon/DiagonalOrthantLDA)

![YourKit](https://www.yourkit.com/images/yklogo.png)

# DOLDA

Supervised LDA using DO-Probit

This is the repo for our Diagonal Orthant Latent Dirichlet Allocation (DOLDA) implementation described in the article [arXiv:1602.00260](https://arxiv.org/abs/1602.00260 "arXiv:1602.00260"): 

```
   @unpublished{MagnussonJonsson2016,
      author = "Måns Magnusson, Leif Jonsson, Mattias Villani",
      title = "DOLDA - a regularized supervised topic model for high-dimensional multi-class regression",
      note = "http://github.com/lejon/DiagonalOrthantLDA",
      year = 2016}
```
The toolkit is Open Source Software, and is released under the Common Public License. You are welcome to use the code under the terms of the license for research or commercial purposes, however please acknowledge its use with a citation:
  Magnusson, Jonsson, Villani.  "DOLDA - a regularized supervised topic model for high-dimensional multi-class regression"
  
For very large datasets you might need to add the -Xmx60g flag to Java

Please remember that this is a research prototype and the standard disclaimers apply.
You will see printouts during unit tests, commented out code, old stuff not cleaned out yet etc.

## Repo extras

- extralibs
  Contains external dependencies which are not available from Maven Central or any other public repo yet.
  
## Installation

1. Install Apache Maven and run:

```mvn package```

in a shell of your choice.

## Example Run
```java -Xmx10g -cp target/DOLDA-1.6.0.jar  xyz.lejon.runnables.DOLDAClassificationDistribution -normalize --run_cfg=src/main/resources/configuration/films.cfg```

Acknowledgements
----------------
I'm a very satisfied user of the YourKit profiler. A Great product with great support. It has been sucessfully used for profiling in this project.

![YourKit](https://www.yourkit.com/images/yklogo.png)

YourKit supports open source projects with its full-featured Java Profiler.
YourKit, LLC is the creator of [YourKit Java Profiler](https://www.yourkit.com/java/profiler/)
and [YourKit .NET Profiler](https://www.yourkit.com/.net/profiler/),
innovative and intelligent tools for profiling Java and .NET applications.
