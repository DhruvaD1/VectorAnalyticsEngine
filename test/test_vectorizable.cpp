void test_simple_vectorizable() {
    int a[100], b[100], c[100];
    
    for (int i = 0; i < 100; ++i) {
        a[i] = b[i] + c[i];
    }
}

void test_vectorizable_with_constant() {
    float data[1000];
    
    for (int i = 0; i < 1000; i++) {
        data[i] = data[i] * 2.0f;
    }
}

void test_vectorizable_multiple_arrays() {
    double x[500], y[500], z[500], w[500];
    
    for (int j = 0; j < 500; ++j) {
        x[j] = y[j] * z[j] + w[j];
    }
}
