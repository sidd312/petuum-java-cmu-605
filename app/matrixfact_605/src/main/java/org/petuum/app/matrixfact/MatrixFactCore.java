package org.petuum.app.matrixfact;

import org.petuum.app.matrixfact.Rating;
import org.petuum.app.matrixfact.LossRecorder;
import org.petuum.ps.row.double_.DenseDoubleRowUpdate;
import org.petuum.ps.row.double_.DoubleRow;
import org.petuum.ps.row.double_.DoubleRowUpdate;
import org.petuum.ps.table.DoubleTable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;

public class MatrixFactCore {
  private static final Logger logger = LoggerFactory
      .getLogger(MatrixFactCore.class);

  // Perform a single SGD on a rating and update LTable and RTable
  // accordingly.
  public static void sgdOneRating(Rating r, double learningRate,
      DoubleTable LTable, DoubleTable RTable, int K, double lambda) {
    int row = r.userId;
    int col = r.prodId;
    DoubleRow lRow = LTable.get(row);
    DoubleRow rRow = RTable.get(col);
    double ni = lRow.get(K);
    double mj = rRow.get(K);

    Double eij = r.rating - dotProduct(lRow, rRow, K);

    DoubleRowUpdate rowUpdates = new DenseDoubleRowUpdate(K);
    DoubleRowUpdate colUpdates = new DenseDoubleRowUpdate(K);

    for (int jindex = 0; jindex < K; jindex++) {
      rowUpdates.setUpdate(jindex, 2 * learningRate
          * (eij * rRow.get(jindex) - lambda * lRow.get(jindex) / ni));
      colUpdates.setUpdate(jindex, 2 * learningRate
          * (eij * lRow.get(jindex) - lambda * rRow.get(jindex) / mj));
    }
    LTable.batchInc(row, rowUpdates);
    RTable.batchInc(col, colUpdates);
  }

  // Evaluate square loss on entries [elemBegin, elemEnd), and L2-loss on of
  // row [LRowBegin, LRowEnd) of LTable, [RRowBegin, RRowEnd) of Rtable.
  // Note the interval does not include LRowEnd and RRowEnd. Record the loss to
  // lossRecorder.
  public static void evaluateLoss(ArrayList<Rating> ratings, int ithEval,
      int elemBegin, int elemEnd, DoubleTable LTable, DoubleTable RTable,
      int LRowBegin, int LRowEnd, int RRowBegin, int RRowEnd,
      LossRecorder lossRecorder, int K, double lambda) {

    double sqLoss = 0;
    double totalLoss = 0;

    for (int index = LRowBegin; index < LRowEnd; index++) {
      DoubleRow temp = LTable.get(index);
      for (int jindex = 0; jindex < K; jindex++) {
        totalLoss += Math.pow(temp.get(jindex), 2);
      }
    }

    for (int i = RRowBegin; i < RRowEnd; i++) {
      DoubleRow rtemp = RTable.get(i);
      for (int j = 0; j < K; j++) {
        totalLoss += Math.pow(rtemp.get(j), 2);
      }
    }

    totalLoss *= lambda;

    // calculation of square loss
    for (int index = elemBegin; index < elemEnd; index++) {
      DoubleRow ltemp = LTable.get(index);
      DoubleRow rtemp = RTable.get(index);
      sqLoss += Math.pow(
          ratings.get(index).rating - dotProduct(ltemp, rtemp, K), 2);
    }
    totalLoss += sqLoss;

    lossRecorder.incLoss(ithEval, "SquareLoss", sqLoss);
    lossRecorder.incLoss(ithEval, "FullLoss", totalLoss);
    lossRecorder.incLoss(ithEval, "NumSamples", elemEnd - elemBegin);
  }

  private static double dotProduct(DoubleRow lRow, DoubleRow rRow, int K) {
    double product = 0D;
    for (int index = 0; index < K; index++) {
      product += lRow.get(index) * rRow.get(index);
    }
    return product;
  }
}
