package xyz.lejon;

import org.junit.experimental.categories.Categories;
import org.junit.runner.RunWith;
import org.junit.runners.Suite.SuiteClasses;

@RunWith(Categories.class)
@Categories.IncludeCategory(MarkerIFParallelTest.class)
@SuiteClasses( AllTestsSuites.class )
public class ParallelTestsSuite {}
