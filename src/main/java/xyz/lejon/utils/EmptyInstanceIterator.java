package xyz.lejon.utils;

import java.util.Iterator;

import cc.mallet.types.Alphabet;
import cc.mallet.types.Instance;
import cc.mallet.types.LabelAlphabet;

public class EmptyInstanceIterator implements Iterator<Instance> {
	String [] labels;
	String [] ids;
	LabelAlphabet targetAlphabet;
	Alphabet dataAlphabet;
	int index;

	public EmptyInstanceIterator (String[] labels, String [] ids, LabelAlphabet targetAlphabet, Alphabet alphabet)
	{
		this.labels = labels;
		this.ids = ids;
		this.targetAlphabet = targetAlphabet;
		this.index = 0;
		dataAlphabet = alphabet;
	}

	public Instance next ()
	{
		String instanceId = ids[index];
		String instanceLabel = labels[index++];
		Instance fakeInstance = new Instance (new SimpleStringData(dataAlphabet, ""), instanceLabel, instanceId, null);
		fakeInstance.setLabeling(targetAlphabet.lookupLabel(instanceLabel));
		return fakeInstance;
	}

	public boolean hasNext ()	{	return index < labels.length;	}

	public void remove () {
		throw new IllegalStateException ("This Iterator<Instance> does not support remove().");
	}
}