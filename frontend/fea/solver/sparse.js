/**
 * Sparse matrix library for FEA solver.
 *
 * Two formats:
 *   COO (Coordinate) - used during assembly; allows duplicate entries.
 *   CSC (Compressed Sparse Column) - used for solving; efficient SpMV.
 *
 * All indices are 0-based. Matrices are square (n x n).
 */

// ---------------------------------------------------------------------------
// COO (Coordinate / Triplet) format
// ---------------------------------------------------------------------------

/**
 * Coordinate (triplet) sparse matrix format.
 * Efficient for incremental assembly: just append entries.
 * Duplicate (row, col) pairs are summed during conversion to CSC.
 */
export class COOMatrix {
  /**
   * @param {number} n - Matrix dimension (n x n square matrix)
   */
  constructor(n) {
    this.n = n;
    /** @type {number[]} Row indices */
    this.rows = [];
    /** @type {number[]} Column indices */
    this.cols = [];
    /** @type {number[]} Values */
    this.vals = [];
  }

  /**
   * Add an entry to the triplet lists.
   * Duplicate (row, col) pairs are accumulated (summed) during toCSC().
   *
   * @param {number} row - Row index (0-based)
   * @param {number} col - Column index (0-based)
   * @param {number} value - Numeric value to add
   */
  addEntry(row, col, value) {
    if (value === 0) return; // skip explicit zeros
    this.rows.push(row);
    this.cols.push(col);
    this.vals.push(value);
  }

  /**
   * Return the number of stored entries (including duplicates).
   * @returns {number}
   */
  get nnz() {
    return this.rows.length;
  }

  /**
   * Convert this COO matrix to CSC format, summing duplicate entries.
   *
   * Algorithm:
   *   1. Count entries per column.
   *   2. Build column pointer array.
   *   3. Place entries into CSC arrays, sorting by column.
   *   4. Within each column, sort by row and merge duplicates.
   *
   * @returns {CSCMatrix}
   */
  toCSC() {
    const n = this.n;
    const nnzRaw = this.rows.length;

    if (nnzRaw === 0) {
      // Empty matrix
      const colPtr = new Int32Array(n + 1); // all zeros
      const rowIdx = new Int32Array(0);
      const values = new Float64Array(0);
      return new CSCMatrix(n, colPtr, rowIdx, values);
    }

    // Step 1: count entries per column
    const colCount = new Int32Array(n);
    for (let k = 0; k < nnzRaw; k++) {
      colCount[this.cols[k]]++;
    }

    // Step 2: build column pointers (cumulative sum)
    const colPtr = new Int32Array(n + 1);
    for (let j = 0; j < n; j++) {
      colPtr[j + 1] = colPtr[j] + colCount[j];
    }

    // Step 3: place entries into position
    const rowIdxTemp = new Int32Array(nnzRaw);
    const valuesTemp = new Float64Array(nnzRaw);
    const cursor = new Int32Array(n); // current insert position per column
    for (let j = 0; j < n; j++) {
      cursor[j] = colPtr[j];
    }

    for (let k = 0; k < nnzRaw; k++) {
      const col = this.cols[k];
      const pos = cursor[col];
      rowIdxTemp[pos] = this.rows[k];
      valuesTemp[pos] = this.vals[k];
      cursor[col]++;
    }

    // Step 4: within each column, sort by row index and merge duplicates
    const finalRows = [];
    const finalVals = [];
    const newColPtr = new Int32Array(n + 1);

    for (let j = 0; j < n; j++) {
      const start = colPtr[j];
      const end = colPtr[j + 1];

      // Extract column entries
      const entries = [];
      for (let p = start; p < end; p++) {
        entries.push({ row: rowIdxTemp[p], val: valuesTemp[p] });
      }

      // Sort by row index
      entries.sort((a, b) => a.row - b.row);

      // Merge duplicates (sum values with same row index)
      for (let p = 0; p < entries.length; p++) {
        const row = entries[p].row;
        let val = entries[p].val;
        while (p + 1 < entries.length && entries[p + 1].row === row) {
          p++;
          val += entries[p].val;
        }
        if (val !== 0) {
          finalRows.push(row);
          finalVals.push(val);
        }
      }

      newColPtr[j + 1] = finalRows.length;
    }

    return new CSCMatrix(
      n,
      newColPtr,
      new Int32Array(finalRows),
      new Float64Array(finalVals)
    );
  }
}

// ---------------------------------------------------------------------------
// CSC (Compressed Sparse Column) format
// ---------------------------------------------------------------------------

/**
 * Compressed Sparse Column matrix format.
 * Efficient for column-wise access and sparse matrix-vector multiply.
 *
 * Storage:
 *   colPtr[j]..colPtr[j+1]-1 gives the range of entries in column j.
 *   rowIdx[p] is the row index of entry p.
 *   values[p] is the numeric value of entry p.
 */
export class CSCMatrix {
  /**
   * @param {number}       n       - Matrix dimension (n x n)
   * @param {Int32Array}    colPtr  - Column pointers, length n+1
   * @param {Int32Array}    rowIdx  - Row indices, length nnz
   * @param {Float64Array}  values  - Numeric values, length nnz
   */
  constructor(n, colPtr, rowIdx, values) {
    this.n = n;
    this.colPtr = colPtr;
    this.rowIdx = rowIdx;
    this.values = values;
  }

  /**
   * Number of stored non-zero entries.
   * @returns {number}
   */
  get nnz() {
    return this.values.length;
  }

  /**
   * Access element (i, j). Returns 0 if not stored.
   * Uses binary search within the column for efficiency.
   *
   * @param {number} i - Row index (0-based)
   * @param {number} j - Column index (0-based)
   * @returns {number}
   */
  get(i, j) {
    const start = this.colPtr[j];
    const end = this.colPtr[j + 1];

    // Binary search for row i in column j
    let lo = start;
    let hi = end - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const r = this.rowIdx[mid];
      if (r === i) return this.values[mid];
      if (r < i) lo = mid + 1;
      else hi = mid - 1;
    }
    return 0;
  }

  /**
   * Sparse matrix-vector multiply: y = A * x.
   *
   * @param {Float64Array} x - Input vector of length n
   * @returns {Float64Array} y - Result vector of length n
   */
  multiplyVector(x) {
    const n = this.n;
    const y = new Float64Array(n);

    for (let j = 0; j < n; j++) {
      const xj = x[j];
      if (xj === 0) continue; // skip zero columns
      const start = this.colPtr[j];
      const end = this.colPtr[j + 1];
      for (let p = start; p < end; p++) {
        y[this.rowIdx[p]] += this.values[p] * xj;
      }
    }

    return y;
  }

  /**
   * Create a CSC matrix from a COO matrix, treating it as symmetric.
   * Only the upper triangle (including diagonal) of the input is expected,
   * but both (i,j) and (j,i) entries are stored in the output.
   * Duplicates are summed.
   *
   * @param {COOMatrix} coo - Input COO matrix (upper triangle)
   * @returns {CSCMatrix}
   */
  static fromCOOSymmetric(coo) {
    // Expand: for each off-diagonal entry (i,j) with i != j,
    // add both (i,j) and (j,i).
    const expanded = new COOMatrix(coo.n);
    const nnz = coo.rows.length;
    for (let k = 0; k < nnz; k++) {
      const r = coo.rows[k];
      const c = coo.cols[k];
      const v = coo.vals[k];
      expanded.addEntry(r, c, v);
      if (r !== c) {
        expanded.addEntry(c, r, v);
      }
    }
    return expanded.toCSC();
  }

  /**
   * Create a deep copy of this CSC matrix.
   * @returns {CSCMatrix}
   */
  clone() {
    return new CSCMatrix(
      this.n,
      new Int32Array(this.colPtr),
      new Int32Array(this.rowIdx),
      new Float64Array(this.values)
    );
  }

  /**
   * Convert this CSC matrix to a dense row-major Float64Array.
   * Only practical for small matrices (debugging / dense Cholesky).
   *
   * @returns {Float64Array} Row-major dense matrix of size n*n
   */
  toDense() {
    const n = this.n;
    const dense = new Float64Array(n * n);
    for (let j = 0; j < n; j++) {
      const start = this.colPtr[j];
      const end = this.colPtr[j + 1];
      for (let p = start; p < end; p++) {
        dense[this.rowIdx[p] * n + j] = this.values[p];
      }
    }
    return dense;
  }
}
