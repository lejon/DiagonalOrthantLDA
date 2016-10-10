package xyz.lejon.bayes.models.dolda;

import cc.mallet.topics.LDADocSamplingContext;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.LabelSequence;

class LDADocSamplingContextTest implements LDADocSamplingContext {
	int docId = -1;
	double [][] phi;
	double [][] betas; 
	int ks;
	double alpha;
	double [][] xs;
	double [][] Ast;
	int [] ws; 
	int [] zs; 

	public LDADocSamplingContextTest(int docId,
			double[][] phi, double[][] betas, int ks, double alpha, double[][] xs, double[][] ast, int [] words, int [] zs) {
		super();
		this.docId = docId;
		this.phi = phi;
		this.betas = betas;
		this.ks = ks;
		this.alpha = alpha;
		this.xs = xs;
		Ast = ast;
		ws = words;
		this.zs = zs; 
	}
	public int getDocId() {
		return docId;
	}
	public void setDocId(int docId) {
		this.docId = docId;
	}
	public double[][] getPhi() {
		return phi;
	}
	public int getKs() {
		return ks;
	}
	public double getAlpha() {
		return alpha;
	}
	public double[][] getBetas() {
		return betas;
	}
	public double[][] getX() {
		return xs;
	}
	public double[][] getAst() {
		return Ast;
	}
	public int[] getWs() {
		return ws;
	}
	public int[] getZs() {
		return zs;
	}
	@Override
	public FeatureSequence getTokens() {
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	public void setTokens(FeatureSequence tokens) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public LabelSequence getTopics() {
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	public void setTopics(LabelSequence topics) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public int getMyBatch() {
		// TODO Auto-generated method stub
		return 0;
	}
	@Override
	public void setMyBatch(int myBatch) {
		// TODO Auto-generated method stub
		
	}
}