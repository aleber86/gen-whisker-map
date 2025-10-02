
void Flat_to_zero(double *A_in, int size, double tollerance){
	//If element is < tollerance, element dropped to 0.

	for(int index = 0; index < size; index++){
		if(fabs(A_in[index])<tollerance) A_in[index] = (double) 0.;
	}

}


void vector_to_matrix(double3 v, double *Out){
    Out[0] = v.x*v.x; Out[1] = v.x*v.y; Out[2] = v.x*v.z;
    Out[3] = v.y*v.x; Out[4] = v.y*v.y; Out[5] = v.y*v.z;
    Out[6] = v.z*v.x; Out[7] = v.z*v.y; Out[8] = v.z*v.z;
    
}


double3 Matrix_product_vector(double *A, double3 vector){
    double3 A_1, A_2, A_3;
    double3 res;
    
    A_1 = (double3) (A[0], A[1], A[2]);
    A_2 = (double3) (A[3], A[4], A[5]);
    A_3 = (double3) (A[6], A[7], A[8]);
    

    res.x = dot(A_1, vector);
    res.y = dot(A_2, vector);
    res.z = dot(A_3, vector);

    return res;
}

void Matrix_product(double *A, double *B, double *C){
    double3 A_1, A_2, A_3;
    double3 B_1, B_2, B_3;

    A_1 = (double3) (A[0], A[1], A[2]);
    A_2 = (double3) (A[3], A[4], A[5]);
    A_3 = (double3) (A[6], A[7], A[8]);

    B_1 = (double3) (B[0], B[3], B[6]);
    B_2 = (double3) (B[1], B[3 + 1], B[6 + 1]);
    B_3 = (double3) (B[2], B[3 + 2], B[6 + 2]);

    C[0] = dot(A_1,B_1); C[1] = dot(A_1,B_2); C[2] = dot(A_1,B_3);
    C[3] = dot(A_2,B_1); C[4] = dot(A_2,B_2); C[5] = dot(A_2,B_3);
    C[6] = dot(A_3,B_1); C[7] = dot(A_3,B_2); C[8] = dot(A_3,B_3);
}

void Transpose(double *A, double *T){
    double3 A_1, A_2, A_3;

    A_1 = (double3)(A[0], A[1], A[2]);
    A_2 = (double3)(A[3], A[4], A[5]);
    A_3 = (double3)(A[6], A[7], A[8]);

    T[0] = A_1.x; T[3] = A_1.y; T[6] = A_1.z;
    T[1] = A_2.x; T[4] = A_2.y; T[7] = A_2.z;
    T[2] = A_3.x; T[5] = A_3.y; T[8] = A_3.z;

}

void HQR(double *A, double *Q, double *R){
	double3 x, e1, u1, v;
	double aleph, v_v_T[9], H[9];
	double I[9]  = {1., 0., 0.,
		 0., 1., 0.,
		 0., 0., 1.};

	//First reflection ->

	x = (double3)(A[0], A[3], A[6]);
	e1 = (double3)(1., 0., 0.);
	aleph = -sign(x.x) * length(x);
	u1 = x - aleph * e1;
	v = u1 / length(u1);
	//Outter product v * v^T
	vector_to_matrix(v, v_v_T);
	for(int index = 0; index < 9; index++){
		H[index] = I[index] - 2*v_v_T[index];
		Q[index] = H[index];	
	}					
				
	Matrix_product(H, A, R);
	
	//Second Reflection ->
	
	x = (double3)(0., R[4], R[7]);
	e1 = (double3)(0.,1.,0.);
	aleph = -sign(x.y)*length(x);
	u1 = x - aleph*e1;
	v = u1 / length(u1);
	vector_to_matrix(v, v_v_T);
	for(int index=0; index < 9; index++){
		H[index] = I[index] - 2*v_v_T[index];
	}
	Matrix_product(H,R,R);
	Transpose(H, v_v_T);
	Matrix_product(Q,v_v_T, Q);
	
    for(int index = 0; index < 9; index++){

	H[index] = 0.;
    }
	H[0] = 1.; H[4] = 1.; H[8] = -1.;

	Matrix_product(H, R, R);
	Transpose(H,v_v_T);
	Matrix_product(Q,v_v_T, Q);
    Flat_to_zero(Q, 9, 1.e-15);
    Flat_to_zero(R, 9, 1.e-15);
	
}

