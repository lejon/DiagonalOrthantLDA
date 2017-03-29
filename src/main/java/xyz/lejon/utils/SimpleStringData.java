package xyz.lejon.utils;

import cc.mallet.types.Alphabet;
import cc.mallet.types.AlphabetCarrying;

public class SimpleStringData implements AlphabetCarrying {
	Alphabet alphabet;
	String data;

	public SimpleStringData(Alphabet alphabet, String data) {
		super();
		this.alphabet = alphabet;
		this.data = data;
	}

	@Override
	public Alphabet getAlphabet() {
		return alphabet;
	}

	@Override
	public Alphabet[] getAlphabets() {
		Alphabet [] as = new Alphabet[1];
		as[0] = alphabet;
		return as;
	}

	@Override 
	public String toString() {
		return data;
	}
}