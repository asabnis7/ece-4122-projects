#include <iostream>
#include <vector>
#include <sstream>
#include "simple_string.h"

#define ARRAY

#ifdef ARRAY
#include "array.h"
#endif

using std::vector;
using std::ostringstream;


#ifdef ARRAY

//simple test with default construct and add elements with push_back and push_front.  Retrieve elements with
//only the indexing operator

void test1() {
    std::cout << "starting test1" << std::endl;
    array<simple_string> v;
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string back " << i;
        v.push_back(simple_string(oss.str().c_str()));
    }
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string front " << i;
        v.push_front(simple_string(oss.str().c_str()));
    }
    std::cout << "test1 results" << std::endl;
    for(int i = 0; i < v.length(); ++i) {
        std::cout << v[i] << std::endl;
    }
}

// Just create a vector with default constructor and add elements with Push_Back and Push_Front.  Then retrieve
// with only the Back() function and Pop_Back.  Also tests "Empty"

void test2() {
    std::cout << "starting test2" << std::endl;
    array<simple_string> v;
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string back " << i;
        v.push_back(simple_string(oss.str().c_str()));
    }
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string front " << i;
        v.push_front(simple_string(oss.str().c_str()));
    }
    std::cout << "test2 results" << std::endl;

    while(!v.empty()) {
        simple_string st = v.back();
        v.pop_back();
        std::cout << st << std::endl;
    }
}

// Just create a vector with default constructor and add elements with Push_Back and Push_Front.  Then retrieve
// with only the front() function and Pop_Front.  Also tests "Empty"
void test3() {
    std::cout << "starting test3" << std::endl;
    array<simple_string> v;
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string back " << i;
        v.push_back(simple_string(oss.str().c_str()));
    }
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string front " << i;
        v.push_front(simple_string(oss.str().c_str()));
    }

    std::cout << "test3 results" << std::endl;
    while(!v.empty()) {
        simple_string st = v.front();
        v.pop_front();
        std::cout << st << std::endl;
    }
}

void test4() {
    std::cout << "starting test4" << std::endl;
    array<simple_string> v;
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string back " << i;
        v.push_back(simple_string(oss.str().c_str()));
    }
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string front " << i;
        v.push_front(simple_string(oss.str().c_str()));
    }

    v.clear();
    std::cout << "test4 results" << std::endl;
    for(size_t i = 0; i < v.length(); ++i) {
        std::cout << v[i] << std::endl;
    }
}

void test5() {
    std::cout << "starting test5" << std::endl;
    array<simple_string> v;
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string back " << i;
        v.push_back(simple_string(oss.str().c_str()));
    }
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string front " << i;
        v.push_front(simple_string(oss.str().c_str()));
    }
    std::cout << "test5 results" << std::endl;
    array_iterator<simple_string> it = v.begin();
    while(it != v.end()) {
        std::cout << *it++ << std::endl;
    }
}

void test6() {
    std::cout << "starting test6" << std::endl;
    array<simple_string> v;
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string back " << i;
        v.push_back(simple_string(oss.str().c_str()));
    }
    for(int i = 0; i < v.length(); ++i) {
        ostringstream oss;
        oss << "Hello from string replaced " << i;
        v[i] = simple_string(oss.str().c_str());
    }

    std::cout << "test6 results" << std::endl;
    for(size_t i = 0; i < v.length(); ++i) {
        std::cout << v[i] << std::endl;
    }
}

void test7() {
    std::cout << "starting test7" << std::endl;
    array<simple_string> v;
    for (int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string back " << i;
        v.push_back(simple_string(oss.str().c_str()));
    }
    for (int i = 0; i < v.length(); ++i) {
        ostringstream oss;
        oss << "Hello from string replaced " << i;
        v[i] = simple_string(oss.str().c_str());
    }

    array_iterator<simple_string> it;
    array_iterator<simple_string> it1 = v.end();
 /*   size_t i = 0;
    for (it = v.begin(); it != v.end(); ++it) {
        if (++i == 25) it1 = it;
    }
    v.insert(simple_string("Inserted element before 25th element (item #24)"), it1);
*/
    std::cout << "test7 results" << std::endl;
    array_iterator<simple_string> it2 = v.end();
    size_t i2 = 0;
    for (array_iterator<simple_string> it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << std::endl;
        if (++i2 == 10) it2 = it;
    }

    v.erase(it2);
    std::cout << "test 7 results again, should be missing 10th element (item #9)" << std::endl;
    for (array_iterator<simple_string> it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << std::endl;
    }
}

void test8() {
    std::cout << "starting test8" << std::endl;
    simple_string::initialize_counts();
    array<simple_string> v (20);

    std::cout << "test 8 results" << std::endl;
    std::cout << "Number of constructors called in simple string: " << simple_string::get_default_count() << ", " <<
            simple_string::get_create_count() << std::endl;
}

void test9() {
    std::cout << "starting test9" << std::endl;
    simple_string::initialize_counts();
    array<simple_string> v;
    v.reserve(20);

    std::cout << "test 9 results" << std::endl;
    std::cout << "Number of constructors called in simple string: " << simple_string::get_default_count() << ", " <<
              simple_string::get_create_count() << std::endl;
}

void test10() {
    std::cout << "starting test10" << std::endl;
    array<simple_string> v(10, simple_string("Woot"));
    std::cout << "test 10 results" << std::endl;
    for(size_t i = 0; i < v.length(); ++i) {
        std::cout << v[i] << std::endl;
    }
}

void test11() {
    std::cout << "starting test11" << std::endl;
    simple_string a("Goober");
    simple_string b("Gabber");
    simple_string c("Gupper");

    array<simple_string> v;
    v.push_back(a);
    v.push_back(b);
    v.push_back(c);
    simple_string::initialize_counts();
    v.erase(v.begin());
    //should be moving the others to the front, not copying
    std::cout << "test11 results" << std::endl;
    std::cout << "Number of move assignments " << simple_string::get_move_assign() << std::endl;
}

void test12() {
    std::cout << "starting test12" << std::endl;

    simple_string a("Goober");
    simple_string b("Gabber");
    simple_string c("Gupper");

    array<simple_string> v({a, b, c});
    std::cout << "test12 results" << std::endl;
    for(int i = 0; i < v.length(); ++i) {
        std::cout << v[i] << std::endl;
    }
}
#else

template<typename T>
using array = std::vector<T>;

//simple test with default construct and add elements with push_back and push_front.  Retrieve elements with
//only the indexing operator

void test1() {
    std::cout << "starting test1" << std::endl;
    array<simple_string> v;
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string back " << i;
        v.push_back(simple_string(oss.str().c_str()));
    }
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string front " << i;
        v.insert(v.begin(), simple_string(oss.str().c_str()));
    }
    std::cout << "test1 results" << std::endl;
    for(int i = 0; i < v.size(); ++i) {
        std::cout << v[i] << std::endl;
    }
}

// Just create a vector with default constructor and add elements with Push_Back and Push_Front.  Then retrieve
// with only the Back() function and Pop_Back.  Also tests "Empty"

void test2() {
    std::cout << "starting test2" << std::endl;
    array<simple_string> v;
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string back " << i;
        v.push_back(simple_string(oss.str().c_str()));
    }
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string front " << i;
        v.insert(v.begin(), simple_string(oss.str().c_str()));
    }
    std::cout << "test2 results" << std::endl;

    while(!v.empty()) {
        simple_string st = v.back();
        v.pop_back();
        std::cout << st << std::endl;
    }
}

// Just create a vector with default constructor and add elements with Push_Back and Push_Front.  Then retrieve
// with only the front() function and Pop_Front.  Also tests "Empty"
void test3() {
    std::cout << "starting test3" << std::endl;
    array<simple_string> v;
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string back " << i;
        v.push_back(simple_string(oss.str().c_str()));
    }
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string front " << i;
        v.insert(v.begin(), simple_string(oss.str().c_str()));
    }

    std::cout << "test3 results" << std::endl;
    while(!v.empty()) {
        simple_string st = v.front();
        v.erase(v.begin());
        std::cout << st << std::endl;
    }
}

void test4() {
    std::cout << "starting test4" << std::endl;
    array<simple_string> v;
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string back " << i;
        v.push_back(simple_string(oss.str().c_str()));
    }
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string front " << i;
        v.insert(v.begin(), simple_string(oss.str().c_str()));
    }

    v.clear();
    std::cout << "test4 results" << std::endl;
    for(size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i] << std::endl;
    }
}

void test5() {
    std::cout << "starting test5" << std::endl;
    array<simple_string> v;
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string back " << i;
        v.push_back(simple_string(oss.str().c_str()));
    }
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string front " << i;
        v.insert(v.begin(), simple_string(oss.str().c_str()));
    }
    std::cout << "test5 results" << std::endl;
    auto it = v.begin();
    while(it != v.end()) {
        std::cout << *it++ << std::endl;
    }
}

void test6() {
    std::cout << "starting test6" << std::endl;
    array<simple_string> v;
    for(int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string back " << i;
        v.push_back(simple_string(oss.str().c_str()));
    }
    for(int i = 0; i < v.size(); ++i) {
        ostringstream oss;
        oss << "Hello from string replaced " << i;
        v[i] = simple_string(oss.str().c_str());
    }

    std::cout << "test6 results" << std::endl;
    for(size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i] << std::endl;
    }
}

void test7() {
    std::cout << "starting test7" << std::endl;
    array<simple_string> v;
    for (int i = 0; i < 50; ++i) {
        ostringstream oss;
        oss << "Hello from string back " << i;
        v.push_back(simple_string(oss.str().c_str()));
    }
    for (int i = 0; i < v.size(); ++i) {
        ostringstream oss;
        oss << "Hello from string replaced " << i;
        v[i] = simple_string(oss.str().c_str());
    }

    vector<simple_string>::iterator it;
    vector<simple_string>::iterator it1 = v.end();
    size_t i = 0;
    for (it = v.begin(); it != v.end(); ++it) {
        if (++i == 25) it1 = it;
    }
    v.insert(it1, simple_string("Inserted element before 25th element (item #24)"));

    std::cout << "test7 results" << std::endl;
    vector<simple_string>::iterator it2 = v.end();
    size_t i2 = 0;
    for (vector<simple_string>::iterator it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << std::endl;
        if (++i2 == 10) it2 = it;
    }

    v.erase(it2);
    std::cout << "test 7 results again, should be missing 10th element (item #9)" << std::endl;
    for (vector<simple_string>::iterator it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << std::endl;
    }
}

void test8() {
    std::cout << "starting test8" << std::endl;
    simple_string::initialize_counts();
    array<simple_string> v(20);

    std::cout << "test 8 results" << std::endl;
    std::cout << "Number of constructors called in simple string: " << simple_string::get_default_count() << ", " <<
              simple_string::get_create_count() << std::endl;
}

void test9() {
    std::cout << "starting test9" << std::endl;
    simple_string::initialize_counts();
    array<simple_string> v;
    v.reserve(20);

    std::cout << "test 9 results" << std::endl;
    std::cout << "Number of constructors called in simple string: " << simple_string::get_default_count() << ", " <<
              simple_string::get_create_count() << std::endl;
}

void test10() {
    std::cout << "starting test10" << std::endl;
    array<simple_string> v(10, simple_string("Woot"));
    std::cout << "test 10 results" << std::endl;
    for(size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i] << std::endl;
    }
}

void test11() {
    std::cout << "starting test11" << std::endl;
    simple_string a("Goober");
    simple_string b("Gabber");
    simple_string c("Gupper");

    array<simple_string> v;
    v.push_back(a);
    v.push_back(b);
    v.push_back(c);
    simple_string::initialize_counts();
    v.erase(v.begin());
    //should be moving the others to the front, not copying
    std::cout << "test11 results" << std::endl;
    std::cout << "Number of move assignments " << simple_string::get_move_assign() << std::endl;
}

void test12() {
    std::cout << "starting test12" << std::endl;

    simple_string a("Goober");
    simple_string b("Gabber");
    simple_string c("Gupper");

    array<simple_string> v({a, b, c});
    std::cout << "test12 results" << std::endl;
    for(int i = 0; i < v.size(); ++i) {
        std::cout << v[i] << std::endl;
    }
}


#endif
int main() {
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();
    test7();
    test8();
    test9();
    test10();
    test11();
    test12();
}
