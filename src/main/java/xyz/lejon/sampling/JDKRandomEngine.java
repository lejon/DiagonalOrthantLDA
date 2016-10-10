package xyz.lejon.sampling;

import java.util.concurrent.ThreadLocalRandom;

import jdistlib.rng.RandomEngine;

public class JDKRandomEngine extends RandomEngine {

	@Override
	public double nextGaussian() {
		return ThreadLocalRandom.current().nextGaussian();
	}

	@Override
	public double nextDouble() {
		return ThreadLocalRandom.current().nextGaussian();
	}

	@Override
	public float nextFloat() {
		return ThreadLocalRandom.current().nextFloat();
	}

	@Override
	public int nextInt() {
		return ThreadLocalRandom.current().nextInt();
	}

	@Override
	public int nextInt(int n) {
		return ThreadLocalRandom.current().nextInt(n);
	}

	@Override
	public long nextLong() {
		return ThreadLocalRandom.current().nextLong();
	}

	@Override
	public long nextLong(long l) {
		return ThreadLocalRandom.current().nextLong(l);
	}

	@Override
	public RandomEngine clone() {
		return new JDKRandomEngine();
	}

}
