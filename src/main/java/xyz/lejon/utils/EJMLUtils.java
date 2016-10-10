package xyz.lejon.utils;

import org.ejml.alg.dense.linsol.LinearSolverSafe;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.LinearSolverFactory;
import org.ejml.interfaces.linsol.LinearSolver;

public class EJMLUtils {
	
	public static boolean solveCovariace( DenseMatrix64F a , DenseMatrix64F b , DenseMatrix64F x )
    {
        LinearSolver<DenseMatrix64F> solver = LinearSolverFactory.symmPosDef(a.numCols);

        // make sure the inputs 'a' and 'b' are not modified
        solver = new LinearSolverSafe<DenseMatrix64F>(solver);

        if( !solver.setA(a) )
            return false;

        solver.solve(b,x);
        return true;
    }

}
