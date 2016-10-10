package xyz.lejon;

import org.junit.experimental.categories.Categories;
import org.junit.runner.RunWith;
import org.junit.runners.Suite.SuiteClasses;

@RunWith(Categories.class)
@Categories.ExcludeCategory(MarkerIFSlowTests.class)
@SuiteClasses( AllTestsSuites.class )
public class DOLDAFunctionalSuite {}
