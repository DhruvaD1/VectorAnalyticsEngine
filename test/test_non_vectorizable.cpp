#include <cstdio>
#include <cmath>

void test_non_canonical_loop() {
    int a[100];
    
    for (int i = 1; i < 100; ++i) {
        a[i] = i * 2;
    }
}

void test_non_linear_index() {
    int a[100], b[100];
    
    int func(int x) { return x * x; }
    
    for (int i = 0; i < 100; ++i) {
        a[func(i)] = b[i] * 2;
    }
}

void test_function_call_index() {
    float data[1000];
    
    for (int i = 0; i < 1000; ++i) {
        data[(int)sqrt(i)] = data[i] + 1.0f;
    }
}

void test_unpredictable_branch() {
    int a[100], b[100];
    extern int random_value();
    
    for (int i = 0; i < 100; ++i) {
        if (random_value() > 50) {
            a[i] = b[i] * 2;
        } else {
            a[i] = b[i] + 1;
        }
    }
}

void test_early_exit() {
    int a[100], b[100];
    
    for (int i = 0; i < 100; ++i) {
        a[i] = b[i] + 1;
        if (a[i] > 500) {
            break;
        }
    }
}

void test_goto_statement() {
    int a[100];
    
    for (int i = 0; i < 100; ++i) {
        a[i] = i;
        if (i == 50) {
            goto end_loop;
        }
    }
    
    end_loop:
    return;
}

void test_return_in_loop() {
    int a[100];
    
    for (int i = 0; i < 100; ++i) {
        a[i] = i * 2;
        if (i == 75) {
            return;
        }
    }
}

void test_side_effects() {
    int a[100];
    
    for (int i = 0; i < 100; ++i) {
        a[i] = i;
        printf("Processing element %d\n", i);
    }
}

void test_volatile_access() {
    volatile int *data = nullptr;
    int result[100];
    
    for (int i = 0; i < 100; ++i) {
        result[i] = data[i] * 2;
    }
}

void test_memory_allocation() {
    int *a = new int[100];
    
    for (int i = 0; i < 100; ++i) {
        a[i] = i;
        int *temp = new int(i);
        delete temp;
    }
    
    delete[] a;
}

void test_complex_condition() {
    int a[100], b[100];
    extern int complex_function(int);
    
    for (int i = 0; i < 100; ++i) {
        if (complex_function(i) && (i % 3 == 0) && (a[i] > b[i])) {
            a[i] = b[i] + 5;
        }
    }
}
