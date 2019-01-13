#include <iostream>
#include "array.h"

using namespace std;

int main() {
    
    array<int> a1; // default
    array<int> a2 = {1,2,3,4,5}; // initializer list
    array<int> a3(a2); // copy constructor
    array<int> a4(8); // reserved array
    array<int> a5(5,9); // n copies

    a2.push_front(1); // test push front
    a5.push_back(7); // test push back
    a2.print();
    a5.print();

    a2.pop_back(); // test pop back
    a5.pop_front(); // test pop front
    a2.print();
    a5.print();
    
    cout << "a1 empty? " << a1.empty() << endl; // test empty
    cout << "a5 empty? " << a5.empty() << endl; // test empty

    cout << "a3 length: " << a3.length() << endl; // test length
    cout << "a1 length: " << a1.length() << endl; // test length
    cout << "a5 front element: " << a5.front() << endl; // test front
    cout << "a5 back element: " << a5.back() << endl; // test back
    cout << "a2 fifth [4] element: " << a2[4] << endl; // test element retrieval

    a3.clear(); // test clear
    a3.print();

    array_iterator<int> it1; // test default
    array_iterator<int> it2 = a2.end(); // test back pointer
    array_iterator<int> it3 = a5.begin(); // test front pointer
    
    a2.erase(it2); // test erase
    a2.print();
    a5.insert(12,it3); // test insert
    a5.print();
    it3 = a5.begin();
    it3++; // postdecrement
    ++it3; // predecrement
    a5.erase(it3);
    a5.print();
    a5.insert(45, it3);
    a5.print();
    cout << "it3!=it2 " << (it3 != it2) << endl; // inequality
    cout << "it3==it2 " << (it3 == it2) << endl; // equality
    
    cout << "attempt to obtain iterator to empty array" << endl;
    it2 = a1.begin();
    it2 = a1.end();
    cout << "Passed iterator check" << endl;

    cout << "attempt to pop empty array" << endl;
    a1.pop_front();
    a1.pop_back();
    cout << "passed empty pop" << endl;
}
