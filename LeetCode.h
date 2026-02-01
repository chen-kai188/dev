#pragma once
#include <bits/stdc++.h>
#include "include/rapidxml_ext.h"
#include "include/rapidxml_utils.hpp"
#include <ctype.h>
#include <bitset>
#include <pthread.h>
#include <unistd.h>
#include <map>
#include <iostream>
#include <unordered_map>
#include <sys/stat.h>
#include <condition_variable>
#include <type_traits>
#include <functional>
#include <regex>
#include <ext/pool_allocator.h>
#include <utility>
#include <sys/stat.h>
#include <dirent.h>

using namespace std;
struct ListNode
{
    /* data */
    int val;
    ListNode* next;
    ListNode* rand;
    ListNode():val(0),next(nullptr),rand(nullptr){}
    ListNode(int x) : val(x), next(nullptr),rand(nullptr) {}
    ListNode(ListNode *next):val(0), next(next), rand(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next),rand(nullptr) {}
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

template<class T>
struct Queue //use array
{
    int limit;
    T* arr;
    int size;
    int push_idx;
    int pop_idx;
    
    Queue(int limit)
    {
        arr = new T[limit];
        push_idx = 0;
        pop_idx = 0;
        this->limit = limit;
    }
    
    void push(T val)
    {
        if(size == limit)
        {
            cout << "arr has been" << "\n";
            return;
        }
        size++;
        arr[push_idx] = val;
        push_idx = next_idx(push_idx);
    }
    
    T pop()
    {
        if(size == 0)
        {
            cout << "arr is empty" << "\n";
            return -1;
        }
        size--;
        T val = arr[pop_idx];
        pop_idx = next_idx(pop_idx);
        return val;
    }
    
    bool is_empty()
    {
        return size == 0;
    }
    
    int next_idx(int idx)
    {
        return idx == (limit - 1)? 0:idx++;
    }
};
struct QueueUseList //use list
{
    int length;
    ListNode* front;
    ListNode* rear;
    QueueUseList()
    {
        front = nullptr;
        rear = nullptr;
        length = 0;
    }
    void add(int val)
    {
        ListNode* node = new ListNode(val);
        if(front == nullptr)
        {
            rear = front = node;
        }
        else
        {
            rear->next = node;
            rear = node;
        }
        length++;
    }
    int front_val()
    {
        if(front == nullptr)
        {
            cout << "the Queue is empty" << "\n";
            return -1;
        }
        ListNode* ptr = front;
        front = front->next;
        int val = ptr->val;
        delete ptr;
        ptr = nullptr;
        
        length--;
        return val;
    }
    int top()
    {
        if(front == nullptr)
        {
            cout << "the Queue is empty" << "\n";
            return -1;
        }
        return front->val;
    }
    bool is_empty()
    {
        return front==nullptr;
    }
    
    int size()
    {
        return length;
    }

};
//template<class T>
struct StackUseArray //array
{
    int base;
    int idx;
    int* arr;
    int max_size;
    StackUseArray(int max_size)
    {
        arr = new int(max_size);
        this->max_size = max_size;
        base = 0;
        idx = 0;
    }
    void push(int val)
    {
        if(idx == max_size - 1)
        {
            cout << "stack is full" << "\n";
            return;
        }
        arr[idx] = val;
        idx++;
    }
    int pop()
    {
        if(idx == base)
        {
            cout << "stack is empty" << "\n";
            return -1;
        }
        idx--;
        int val = arr[idx];
        return val;
    }
    bool is_empty()
    {
        return idx == base;
    }
};
struct Stack //use list
{
    ListNode* head;
    int length;
    Stack()
    {
        head = nullptr;
        length = 0;
    }
    void push(int val)
    {
        ListNode* node = new ListNode(val);
        node->next = head;
        head = node;
        length++;
    }
    
    int pop()
    {
        if(head == nullptr)
        {
            cout << "the stack is empty" << "\n";
            return -1;
        }
        int val = head->val;
        ListNode* p = head;
        head = head->next;
        delete p;
        p = nullptr;

        length--;
        return val;
    }
    
    int top()
    {
        if(head == nullptr)
        {
            cout << "the stack is empty!!!" << "\n";
            return -1;
        }
        return head->val;
    }
    
    bool is_empty()
    {
        return head == nullptr;
    }
    
    int Size()
    {
        return length;
    }
};
struct StackMinVal //can get min val
{
    Stack* source;
    Stack* min_val;
    StackMinVal()
    {
        source = new Stack();
        min_val   = new Stack();
    }
    void push(int val)
    {
        source->push(val);
        if(min_val->is_empty())
        {
            min_val->push(val);
        }
        else if(val < get_min_val())
        {
            min_val->push(val);
        }
        else
        {
            int minVal = min_val->top();
            min_val->push(minVal);
        }
    }
    int pop()
    {
        source->pop();
        return min_val->pop();
    }
    int get_min_val()
    {
        return min_val->top();
    }
    int top()
    {
        return source->top();
    }
};
struct QueeueUseStack //use stack
{
    Stack* source;
    Stack* help;
    QueeueUseStack()
    {
        source = new Stack();
        help   = new Stack();
    }

    void add(int val)
    {
        source->push(val);
        add_aid_element();
    }
    int pop()
    {
        int val = help->pop();
        add_aid_element();
        return val;
    }
    int front()
    {
        return help->top();
    }
    void add_aid_element()
    {
        if(help->is_empty())
        {
            while (!source->is_empty())
            {
                help->push(source->pop());
            }
        }
    }
};

struct StackUseQueene //use queue
{
    QueueUseList* source;
    QueueUseList* assist;
    StackUseQueene()
    {
        source = new QueueUseList();
        assist = new QueueUseList();
    }
    void push(int val)
    {
        source->add(val);
    }
    int pop()
    {
        while (source->size() > 1)
        {
            // cout << "debug: " << source->top() << "\n";
            assist->add(source->front_val());
        }
        
        int val = source->front_val();
        
        while (!assist->is_empty())
        {   
            source->add(assist->front_val());   
        }
        return val;
    }
    int top()
    {
        while (source->size() > 1)
        {
            assist->add(source->front_val());
        }
        int val = source->front_val();
        assist->add(val);

        while (!assist->is_empty())
        {   
            source->add(assist->front_val());   
        }
        return val;
    }

    bool is_empty()
    {
        return source->is_empty();
    }

    int size()
    {
        return source->size();
    }
};
class HeapUseArray
{
public:
    HeapUseArray(int limit)
    {
        arr = new int[limit];
        this->limit = limit;
    }
    ~HeapUseArray()
    {
        delete[] arr;
    }

    bool push(int val);
    int pop();
    int length();
    bool is_empty();
private:
    int* arr;
    int size = 0;
    int limit = 0;
};

template<class T>
struct ListNodeUseType
{
    /* data */
    T val;
    ListNodeUseType* next;
    ListNodeUseType* rand;
    ListNodeUseType():val(T()),next(nullptr),rand(nullptr){}
    ListNodeUseType(T x) : val(x), next(nullptr),rand(nullptr) {}
};
template<class T>
class QueueUseListMultiType //use list
{
public:
    QueueUseListMultiType()
    {
        front = nullptr;
        rear = nullptr;
        length = 0;
    }
    void add(T val)
    {
        ListNodeUseType<T>* node = new ListNodeUseType<T>(val);
        if(front == nullptr)
        {
            rear = front = node;
        }
        else
        {
            rear->next = node;
            rear = node;
        }
        length++;
    }
    T front_val()
    {
        if(front == nullptr)
        {
            cout << "the Queue is empty" << "\n";
            return T();
        }
        ListNodeUseType<T>* ptr = front;
        front = front->next;
        T val = ptr->val;
        delete ptr;
        ptr = nullptr;
        
        length--;
        return val;
    }
    T top() const
    {
        if(front == nullptr)
        {
            cout << "the Queue is empty" << "\n";
            return T();
        }
        return front->val;
    }
    bool is_empty() const
    {
        return front==nullptr;
    }
    int size() const
    {
        return length;
    }
private:
    int length;
    ListNodeUseType<T>* front;
    ListNodeUseType<T>* rear;
};

class GraghMatrix
{
public:
    GraghMatrix()
    {

    }

};

void test();

ListNode *reverseKGroup(ListNode *head, int k);
ListNode *reverseListNode(ListNode *head);
ListNode *createListNode(initializer_list<int> arr);
int removeElement(vector<int>& nums, int val);
int strStr(string haystack, string needle);

//time out
vector<int> findSubstring(string s, vector<string>& words);
void back_trace(int curr_index, vector<string>& words, set<string>& rets);

vector<int> findSubstring1(string s, vector<string>& words);

void nextPermutation(vector<int>& nums);

int longestValidParentheses(string s);

int search(vector<int>& nums, int target);

int maxProfit(vector<int>& prices);

vector<int> searchRange(vector<int>& nums, int target);

int searchInsert(vector<int>& nums, int target);

bool isValidSudoku(vector<vector<char>>& board);
bool is_unique(vector<vector<char>> &board, int row, int col);

void solveSudoku(vector<vector<char>>& board);
void flip(int i, int j, int digital, vector<int> &rows, vector<int> &cols, vector<vector<int>> &blocks);
void back_trace_sudo(int index, vector<vector<char> > &board, const vector<pair<int, int>> &score, vector<int> &rows, vector<int> &cols, vector<vector<int>> &blocks, bool& is_vaild);

string countAndSay(int n);
vector<vector<int>> combinationSum(vector<int>& candidates, int target);
void back_trace_combination(const vector<int> &candidates, int target, vector<int>& ret, vector<vector<int>> &ans, int index = 0); //include 

vector<vector<int>> combinationSum2(vector<int>& candidates, int target);
void back_trace_combination2(const vector<int> &candidates, int target, vector<int>& ret, vector<vector<int>> &ans, int index = 0); 

int firstMissingPositive(vector<int>& nums);
int trap(vector<int>& height); //43 one
int trap2(vector<int> &height); //43 second

string multiply(string num1, string num2);

vector<vector<int>> permute(vector<int>& nums);
void Recursive_permute(vector<vector<int> >& result, vector<int>& nums, int start);

vector<vector<int>> permuteUnique(vector<int>& nums);
void recursive_permuteUnique(const vector<int>& nums, vector<vector<int> >& result, vector<int>& current, vector<bool>& record_used);

void rotate(vector<vector<int>>& matrix);

vector<vector<string> > groupAnagrams(vector<string>& strs);

double myPow(double x, int n);
vector<vector<string>> solveNQueens(int n);
bool judge_queens(const vector<string>&queens, const int& n, int i, int j);
bool back_queens(vector<vector<string> > &queens, vector<string>& queen, int row, int n);

// void clearBuffer(vector<vector<char> >& buffer);
// void displayBuffer(vector<vector<char> >& buffer);
// void draw3DHeart(vector<vector<char> >& buffer, double angle);
// void rotatePoint(double& x, double& y, double angle);

vector<int> spiralOrder(vector<vector<int>>& matrix);
void generate_jc2bank(const char* path);

int uniquePaths(int m, int n);
int minPathSum(vector<vector<int>>& grid);
bool isNumber(string s);
vector<int> plusOne(vector<int>& digits);
vector<vector<int>> subsets(vector<int>& nums);
void back_subsets(const vector<int>& nums, vector<vector<int> >& ans, vector<int>& result, int index);

ListNode* deleteDuplicates(ListNode* head);
int largestRectangleArea(vector<int>& heights);
int maximalRectangle(vector<vector<char>>& matrix);
bool isScramble(string s1, string s2);
bool isScramble(const string& s2, string& matched, int index);
vector<int> grayCode(int n);
vector<vector<int>> subsetsWithDup(vector<int>& nums);
void back_subsetsWithDup(vector<vector<int> >&ans, vector<int>& ret, const vector<int>& nums, int index);
ListNode* reverseBetween(ListNode* head, int left, int right);
void selection_sort(vector<int>& nums); //选择排序
void bubble_sort(vector<int>& nums);

struct ProbeSignal
{
    int lsb;
    int rsb;
    string signal;
    bool is_bus;
    ProbeSignal(string signal, int lsb, int rsb, bool is_bus)
    {
        this->signal = signal;
        this->lsb    = lsb;
        this->rsb    = rsb;
        this->is_bus = is_bus;
    }
};

void read_probe_list(const char* path);
void deal_with_net(const vector<string>& nets, vector<ProbeSignal>& probe_vec);
int calculate_net_cnt(const vector<ProbeSignal>& probe_vec);
int bit_counts(int N);

void mergeSort(vector<int>& nums);
void process_sort(vector<int>& nums, int left, int right);
void merge(vector<int>& nums, int left, int mid, int right);

void heapSort(vector<int>& arr);
void heap_insert(vector<int>& arr, int index); //down to up
void heap_update(vector<int>& arr, int n, int index); //up to down

struct Student
{
    int age;
    int id;
    string name;
    Student(string name, int age, int id)
    {
        this->name = name;
        this->id   = id;
        this->age  = age;
    }
    bool operator<(const Student& other) const
    {
        if (name != other.name) return name < other.name;
        if (age != other.age) return age < other.age;
        return id < other.id;
    }
};

// template<class T>
struct Node
{
    int pass;
    int end;
    map<char, Node*> next;

    Node()
    {
        pass = 0;
        end  = 0;
    }
    ~Node()
    {
        for (auto iter = next.begin(); iter != next.end(); iter++)
        {
            delete iter->second;
            iter->second = nullptr;
        }
    }
};
class PrefixTree
{
public:
    PrefixTree()
    {
        root = new Node();
    }
    ~PrefixTree();

public:
    void insert(const string& str);
    int search(const string& str);
    int prefix_match(const string& str);
    void delete_element(const string& str);

private:
    void free_node(Node* del_node);

private:
    Node* root;
};

bool palindrome(ListNode* head);
bool palindrome_original_list(ListNode* head);
ListNode* mid_upmid_list(ListNode* head);
ListNode* copy_ListNode(ListNode* head);
ListNode* copy_list_unmap(ListNode* head);
ListNode* check_loop_list(ListNode* head);
ListNode* double_intersection_list(ListNode* head1, ListNode* head2);
ListNode* check_intersection_list(ListNode* head1, ListNode* head2);
ListNode* double_loop_list(ListNode* head1, ListNode* loop1, ListNode* head2, ListNode* loop2);


TreeNode* create_tree_node(const vector<int>& m_list, int null_val = -1);
void preorder_traversal(TreeNode* head);
void midorder_traversal(TreeNode* head);
void backorder_traversal(TreeNode* head);

void preorder_traversal_use_stack(TreeNode* head);
void midorder_traversal_use_stack(TreeNode* head);
void backorder_traversal_use_stack(TreeNode* head);

int max_breadth_tree(TreeNode *head);
int max_breadth_tree_use_map(TreeNode* head);

void serialize_tree_node(TreeNode* head, QueueUseListMultiType<int>& serialized_data);
TreeNode* deserialize_tree_node(QueueUseListMultiType<int>& serialized_data);

bool consecutive_natural_numbers(int N);
bool consecutive_natural_numbers_meter(int N);

void matrix_traversal(const vector<vector<int> >& matrixs);
void printf_matrix(const vector<vector<int> >& matrixs, int x_r, int x_c, int y_r, int y_c, bool read_mode);

void matrix_rotate_traversal(const vector<vector<int> >& matrixs);

struct Meet
{
    int start;
    int end;
    Meet() = default;
    Meet(int start, int end)
    {
        this->start = start;
        this->end   = end;
    }
};
void attain_meet_times(vector<Meet>& meets);
int process_meet_times(const vector<Meet>& meets, int index, int end_time);
int attain_meet_times_use_greedy(vector<Meet>& meets);

void light_place(const string& lights);
int process_light_place(const string &lights, int idx, set<int>& indexs);
int light_place_use_greedy(const string& lights);

struct Program
{
    int cost;
    int profit;

    Program() = default;
    Program(int cost, int profit)
    {
        this->cost = cost;
        this->profit = profit;
    }
};
int attain_max_profit(const vector<Program>& programs, const int& k, int cost);

template<class T>
struct UnionNode
{
    T value;
    UnionNode(T value)
    {
        this->value = value;
    }
};

template<class T>
class UnionSearch
{

public:
    UnionSearch() = default;
    UnionSearch(vector<T> datas);
    bool is_same_union(T A, T B);
    void set_same_union(T A, T B);
    int size();

private:
    UnionNode<T>* find_father(UnionNode<T>* node);

private:
    map<T, UnionNode<T>* > nodes;
    map<UnionNode<T>*, int> set_size;
    map<UnionNode<T>*, UnionNode<T>* > set_parent;

};

struct User
{
    string name;
    string id;
    string account;

    User()=default;
    User(const User& user1)
    {
        this->name = user1.name;
        this->id   = user1.id;
        this->account = user1.account;
    }
    User(User&& user1)
    {
        this->name = move(user1.name);
        this->id   = move(user1.id);
        this->account = move(user1.account);
    }
    User(string name, string id, string account)
    {
        this->name = name;
        this->id   = id;
        this->account = account;
    }
    bool operator<(const User& other) const
    {
        if (id != other.id) return id < other.id;
        if (name != other.name) return name < other.name;
        return account < other.account;
    }
    User& operator=(const User& other)
    {
        if(this != &other)
        {
            this->name = other.name;
            this->id   = other.id;
            this->account = other.account;
        }
        return *this;
    }

    User& operator=(User&& other)
    {
        if(this != &other)
        {
            this->name = move(other.name);
            this->id   = move(other.id);
            this->account = move(other.account);
        }
        return *this;
    }
};

template<class T>
class GrapNode;

template<class T>
class Edge
{
    
public:
    Edge()=default;
    Edge(shared_ptr<GrapNode<T> > from_point, shared_ptr<GrapNode<T> > to_point, double weight);

    double get_weight();
    shared_ptr<GrapNode<T> > get_from_point();
    shared_ptr<GrapNode<T> > get_to_point();

public:
    bool operator<(const Edge& other) const;


private:
    weak_ptr<GrapNode<T> > from_point;
    weak_ptr<GrapNode<T> > to_point;
    double weight;
};

template<class T>
class GrapNode
{
public:
    GrapNode() = default;
    GrapNode(T value);
    T get_value() const;

    bool add_next_node(const shared_ptr<GrapNode<T> >& next);
    bool add_edge(const shared_ptr<Edge<T> >& edge);
    bool add_out_degree();
    bool add_in_degree();

    vector<shared_ptr<Edge<T> > >& get_nearby_edges();
    vector<shared_ptr<GrapNode<T> > >& get_nearby_nodes();

private:
    T value;
    int out_degree;
    int in_degree;
    vector<shared_ptr<GrapNode<T> > >  nearby_nodes;
    vector<shared_ptr<Edge<T> > >  nearby_edges;
};

template<class T>
class Graph
{
public:
    Graph() = default;
    Graph(const vector<tuple<T, T, double> >& datas);
    ~Graph();

    void breadth_first_traversal(T value);
    void depth_first_traversal(T value);
    void topological_sort(vector<T>& ans);
    void minimum_span_subgraph_kruskal(vector<Edge<T> >& update_edges);
    void minimum_span_subgraph_prim(vector<Edge<T> >& update_edges);
    map<shared_ptr<GrapNode<T> >, double> minimum_generation_path(T value);

private:
    vector<T> get_values();

    shared_ptr<GrapNode<T> > find_min_nearbynode(const set<shared_ptr<GrapNode<T> > >& selected_nodes, const map<shared_ptr<GrapNode<T> >, double>& records);

private:
    map<T, shared_ptr<GrapNode<T> > > nodes;
    vector<shared_ptr<Edge<T> > > edges;
};

void hanio(int N, const string& src, const string& dst, const string& aid);
void queens(int N);
int queens_process(vector<int> &matrix, int row);
bool is_valid_queen(const vector<int>& matrix, int i, int j);

int queens_process_use_bit(const int& limit, int col_limit, int left_limit, int right_limit, int row, vector<bitset<8> >& queen_pos);

void min_sticker(const vector<string>& stickers, const string& rest);
int traverse_min_sticker(const vector<vector<int> >& tran_stickers, vector<int> tran_rest, unordered_map<string, int>& dcp);

int numTrees(int n);
int numTrees_use_dp(int n);

vector<TreeNode*> generateTrees(int n);
vector<TreeNode*> traverse_tree(int left, int right);

bool isValidBST(TreeNode* root);
bool traverse_validBST(TreeNode* root, long long min_val, long long max_val);

void recoverTree(TreeNode* root);
void traverse_recoverTree(TreeNode* root, TreeNode* first, TreeNode* second, TreeNode* pre_node);

bool isSameTree(TreeNode* p, TreeNode* q);
bool isSymmetric(TreeNode* root);
bool traverse_isSymmetric(TreeNode* p, TreeNode* q);

vector<vector<int> > levelOrder(TreeNode* root);

vector<vector<int> > zigzagLevelOrder(TreeNode* root);
int maxDepth(TreeNode* root);

TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder);

TreeNode* traverse_buildTree(const vector<int>& preorder, int pre_start, int pre_end, const vector<int>& inorder, int in_start, int in_end, const map<int, int>& in_pos);

TreeNode* buildTree_in_post(vector<int>& inorder, vector<int>& postorder);

bool isBalanced(TreeNode* root);
int minDepth(TreeNode* root);
vector<vector<int>> pathSum(TreeNode* root, int targetSum);
void traverse_hasPathSum(TreeNode* root, int targetSum, vector<int>& one, vector<vector<int> >& ans);

void flatten(TreeNode* root);
int numDistinct(string s, string t);
int traverse_numDistinct(const string& s, const string& t, int index, string& ans);

struct NODE
{
    int val;
    NODE *left;
    NODE *right;
    NODE *next;
    NODE(int val):val(val),left(nullptr),right(nullptr),next(nullptr){}

};
NODE* create_NODE_tree(const vector<int>& data, const int& null_val = -1);
NODE* connect(NODE* root);

int minimumTotal(vector<vector<int>>& triangle);

int maxProfit_II(vector<int>& prices);
int maxProfit_III(vector<int>& prices);
int maxPathSum(TreeNode* root);
int traverse_maxPathSum(TreeNode* root, int& sum);

vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList);
int longestConsecutive(vector<int>& nums);
int sumNumbers(TreeNode* root);
void solve(vector<vector<char>>& board);
vector<vector<string> > partition(string s);
int minCut(string s);

class GraphNode
{
public:
    int val;
    vector<GraphNode*> neighbors;

    GraphNode()
    {
        val = 0;
        neighbors = vector<GraphNode*>();
    }
    GraphNode(int _val)
    {
        val = _val;
        neighbors = vector<GraphNode*>();
    }
    GraphNode(int _val, vector<GraphNode*> _neighbors)
    {
        val = _val;
        neighbors = _neighbors;
    }
};
GraphNode* buildGraph(const vector<vector<int> >& adj);
GraphNode* cloneGraph(GraphNode* node);

int canCompleteCircuit(vector<int>& gas, vector<int>& cost);
int candy(vector<int>& ratings);
bool checkPowersOfThree(int n);
bool isPowerOfFour(int n);
int maximum69Number(int num);
int singleNumber(vector<int>& nums);
int singleNumber2(vector<int>& nums);
double new21Game(int n, int k, int maxPts);
long long zeroFilledSubarray(vector<int>& nums);
int countSquares(vector<vector<int>>& matrix);
class SpecialNode {
public:
    int val;
    SpecialNode* next;
    SpecialNode* random;
    
    SpecialNode(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
SpecialNode* copyRandomList(SpecialNode* head);

struct AverageNode
{

    int p_;
    int t_;
    double gain_;
    AverageNode(int p, int t):p_(p),t_(t)
    {
        gain_ = computeGain(p_, t_);
    }
    static double computeGain(int p, int t)
    {
        return ((double)t-(double)p) / ((double)(t+1.0)*(double)t);
    }
};

double maxAverageRatio(vector<vector<int> >& classes, int extraStudents);
bool wordBreak(string s, vector<string>& wordDict);

template<class T>
class GeneralPinter
{
public:
    GeneralPinter(const GeneralPinter& other) = delete;
    GeneralPinter():ptr(nullptr){}
    GeneralPinter(GeneralPinter&& other) noexcept :ptr(other.ptr)
    {
        other.ptr = nullptr;
    }
    explicit GeneralPinter(const T& val):ptr(new T(val)){}
    explicit GeneralPinter(T* val):ptr(val) {}
    GeneralPinter& operator=(const GeneralPinter& other) = delete;
    GeneralPinter& operator=(GeneralPinter&& other) noexcept
    {
        if(this != &other)
        {
            delete ptr;
            ptr = other.ptr;
            other.ptr=nullptr;
        }
        return *this;
    }
    ~GeneralPinter() noexcept
    {
        delete ptr;
    }

    explicit operator bool() const noexcept
    {
        return ptr != nullptr;
    }

private:
    T* ptr = nullptr;

public:
    T& operator*() const
    {
        return *ptr;
    }
    T* operator->() const
    {
        return ptr;
    }

    T* release() noexcept
    {
        T* temp = ptr;
        ptr = nullptr;
        return temp;
    }
    T* get() noexcept
    {
        return ptr;
    }
};

class Interface
{
public:
    virtual ~Interface() = default;
    virtual void sort(vector<int>& nums) = 0;
};

class InsertSort: public Interface
{
public:
    InsertSort() = default;
    void sort(vector<int>& nums) override;
};

class QuickSort : public Interface
{
public:
    QuickSort() = default;
    void sort(vector<int>& nums) override;
};

class StrategyWay
{
private:
    unique_ptr<Interface> method_;

public:
    StrategyWay() = default;
    void set_execute_method(unique_ptr<Interface> method);

    void execute_method(vector<int>& nums);

};
class Serve;
class Oberve
{
public:
    Oberve(shared_ptr<Serve> serve):server_(serve) {}
    virtual ~Oberve() = default;
    virtual void process() = 0;
protected:
    weak_ptr<Serve> server_;
};

class Serve
{
protected:
    set<shared_ptr<Oberve> > oberves_;

public:
    virtual ~Serve() = default;
    void attach(shared_ptr<Oberve> oberve)
    {
        oberves_.insert(oberve);
    }

    void detach(shared_ptr<Oberve> oberve)
    {
        oberves_.erase(oberve);
    }

    void update()
    {
        for (auto iter=oberves_.begin(); iter!=oberves_.end(); iter++)
        {
            (*iter)->process();
        }
    }
};

class Subject : public Serve
{
private:
    int temperature_;
public:
    explicit Subject() = default;
    virtual ~Subject() = default;
    explicit Subject(int temperature) : temperature_(temperature) {}
    void set_temperature(int temperature)
    {
        if(temperature != temperature_)
        {
            temperature_ = temperature;
            update();
        }
    }
    int get_temperature() const
    {
        return temperature_;
    }
};

class Alarm : public Oberve, public enable_shared_from_this<Alarm>
{
protected:
    int threshold_;

public:
    explicit Alarm(shared_ptr<Serve> serve, int threshold) : Oberve(serve), threshold_(threshold){}
    static shared_ptr<Oberve> create(shared_ptr<Serve> serve, int threshold)
    {
        auto m_alarm = make_shared<Alarm>(serve, threshold);
        auto it = m_alarm->server_.lock();
        it->attach(m_alarm);
        return m_alarm;
    } 
    int get_threshold()
    {
        return threshold_;
    }
    void process() override
    {   
        auto it = server_.lock();
        shared_ptr<Subject> m_sub = dynamic_pointer_cast<Subject>(it);
        if(it && m_sub->get_temperature() > threshold_)
        {
            cout << "temperature: " << m_sub->get_temperature() << " ";
            cout << "the threshold: " << threshold_ << "\n";
        }
    }
};

extern const int THREAD_SIZE;
extern queue<int> works;
extern condition_variable condi_var;
extern mutex mtx;
void produce();
void comsumer();

namespace MemoryAllocateWay
{
#ifdef __cplusplus
extern "C"
{
#endif
    typedef struct MemoryNode{
        MemoryNode* next;
    } MemoryNode;

    struct MeomoryPool
    {   
        uint32_t block_size = 0;
        uint32_t block_chunk_per = 0;
        uint32_t used = 0;
        void** ptr;
        MemoryNode* chunk = NULL;
    };

    MeomoryPool* init_pool(uint32_t block_size, uint32_t pool_size);
    bool allocate_pool(MeomoryPool* pool);
    void free_pool(MeomoryPool* pool);

    void* single_alloc(MeomoryPool* pool);

    bool single_free(MeomoryPool* pool, void* space);

    typedef struct NonFixedMemoryPool //可变内存池的数据变量
    {
        MeomoryPool** fixed_pools;
        uint32_t min_size;
        uint32_t max_size;
        uint32_t unfix_blocks;
        uint32_t* unfix_array;
    } NonFixedMemoryPool;

    uint32_t value_two_pow(uint32_t value); //向上取2的幂次方的值
    NonFixedMemoryPool* init_nonfix_pool(uint32_t min_size, uint32_t max_size, uint32_t capatity); //初始化可变内存池
    void* allocate_nonfixed_pool(NonFixedMemoryPool* pool, uint32_t size); //在给定大小的情况下，分配内存
    void deallocate_nonfixed_pool(NonFixedMemoryPool* pool, void* space, uint32_t size); //回收内存
    void free_nonfixed_pool(NonFixedMemoryPool* pool); //销毁内存池
#ifdef __cplusplus   
}
#endif
};
using namespace MemoryAllocateWay;

template<typename T>
class Allocate
{
public:
    Allocate() = default;
    T* allocate(size_t n)
    {
        if(n == 0) return nullptr;
        T* p = (T*)malloc(n*sizeof(T));
        if(p == nullptr) throw bad_alloc();

        return p;
    }

    void deallocate(T*& p, size_t n)
    {
        free(p);
    }
};

template< typename T, template<typename> class DefaultAllocate = Allocate >
class MyVector
{
private:
    T* ptr = nullptr;
    DefaultAllocate<T> m_vec;

public:
    MyVector(size_t n):ptr(m_vec.allocate(n)){}
    ~MyVector()
    {
        m_vec.deallocate(ptr, 1);
    }
};

template<typename T>
class CustomAllocate
{
public:
    CustomAllocate() = default;
    T* allocate(size_t n)
    {
        if(n == 0) return nullptr;
        return new T();
    }

    void deallocate(T*& p, size_t n)
    {
        delete p;
    }

};

template<template<typename...> class Container, typename...Args>
struct m_size
{
    static constexpr size_t value = sizeof...(Args);
};

class Product
{
public:
    virtual ~Product() = default;
    virtual void process() const = 0;
};

class ProductA : public Product
{
public:
    ~ProductA() = default;
    void operator()() const
    {
        process();
    }
    void process() const override
    {
        cout << "ProductA is running!!!" << "\n";
    }
};

class ProductB : public Product
{
public:
    ~ProductB() = default;
    void operator()() const
    {
        process();
    }
    void process() const override
    {
        cout << "ProductB is running!!!" << "\n";
    }
};

template<typename T>
class CreateFactory
{
public:
    virtual ~CreateFactory() = default;
    static T* create()
    {
        return new T();
    }
};

template<typename T, template<typename> class DefaultFactory = CreateFactory>
class Factory
{
private:
    T* instance = nullptr;
public:
    Factory() = default;
    T* GetInstance()
    {
        if(instance == nullptr)
        {
            instance = DefaultFactory<T>::create();
        }
        return instance;
    }
    void destory_object()
    {
        if(instance == nullptr) return;
        delete instance;
    }
};

template<typename T>
class CustomFactory
{
public:
    static T* create()
    {
        void* ptr = malloc(sizeof(T));
        T* p = new (ptr) T;
        return p;
    }
};

class DrawAble
{
private:
    struct Base
    {
        virtual ~Base() = default;
        virtual void draw() = 0;
    };
    
    template<typename T>
    struct Drived : public Base
    {
        Drived(T callable):callable_(move(callable)){} // Rectangle
        ~Drived() = default;
        void draw() override
        {
            callable_.draw(); // callable_
        }
        T callable_; // Rectangle callable_
    };
    shared_ptr<Base> data_;
public:
    template<typename T>
    DrawAble(T data):data_(make_shared<Drived<T>>(move(data))) {}

    void draw()
    {
        data_->draw();
    }
};
class Rectangle {
public:
    void draw() const { 
        std::cout << "Drawing a rectangle\n"; 
    }
};
class Triangle {
public:
    void draw() const { 
        std::cout << "Drawing a triangle\n"; 
    }
};

namespace ObjectPool
{
struct ObjectMemory
{
    // bool used = false;
    ObjectMemory* next = nullptr;
};
struct ObjectAllocate
{
    uint32_t object_size;
    uint32_t capatity;
    ObjectMemory* memory;
    void* buff;
};
uint32_t round_up_to_align(uint32_t value, uint32_t align_size);

ObjectAllocate* pool_init(uint32_t object_size, uint32_t capatity);
void *pool_allocate(ObjectAllocate *obj_pool);
void pool_deallocate(ObjectAllocate *obj_pool, void *obj);
void destory_pool(ObjectAllocate *obj_pool);
};

template <typename T>
class FixedObject
{
private:
    using Slot = typename aligned_storage<sizeof(T), alignof(T)>::type;
    vector<Slot> requested_memory_;
    vector<uint32_t> requested_size_;
    uint32_t capatity_;
    vector<uint32_t> used;
    
public:
    explicit FixedObject(uint32_t capatity):
        requested_memory_(),
        requested_size_(),
        capatity_(capatity),
        used(capatity, 0)
    {
        requested_memory_.resize(capatity_);

        requested_size_.reserve(capatity_); 
        for (size_t i = 0; i < capatity_; i++)
        {
            requested_size_.push_back(capatity_ - i - 1);
        }
    }
    ~FixedObject()
    {
        capatity_ = 0;
        requested_size_.clear();
        for (int i = capatity_; i < (int)requested_memory_.size(); i++)
        {
            if(!used[i]) continue;
            T* ptr = reinterpret_cast<T*>(&requested_memory_[i]);
            ptr->~T();
        }
        requested_memory_.clear();
        used.clear();
    }

    template<typename...Args>
    T* allocate(Args&&... args)
    {
        if(capatity_ == 0) return nullptr;

        uint32_t idx = requested_size_.back();
        requested_size_.pop_back();

        T* ptr = new (&requested_memory_[idx]) T(forward<Args>(args)...);
        used[idx] = 1;
        return ptr;
    }

    void deallocate(T* ptr)
    {
        if(ptr == nullptr) return;
        Slot* base = requested_memory_.data();
        Slot* dis_t = reinterpret_cast<Slot*>(ptr);

        ptrdiff_t diff = dis_t - base;
        if(diff < 0 || static_cast<uint32_t>(diff) >= capatity_)
        {
            cout << "the ptr is not in pool!!!" << "\n";
            return;
        }

        uint32_t idx = static_cast<uint32_t>(diff);

        ptr->~T();
        used[idx] = 0;
        requested_size_.push_back(idx);
    }

    uint32_t index_of(T* ptr)
    {
        if(ptr == nullptr) return 0;
        Slot* base = requested_memory_.data();
        Slot* dis_t = reinterpret_cast<Slot*>(ptr);
        uint32_t idx = static_cast<uint32_t>(dis_t - base);

        if(idx < 0 || idx >= capatity_)
        {
            cout << "the ptr is not in pool!!!" << "\n";
            return 0;
        }
        return idx;
    }

    uint32_t capatity() const { return capatity_; }
    uint32_t free_count() const { return static_cast<uint32_t>(requested_size_.size()); }

};

template<typename T>
class GrowObjectMemory
{
private:
    using Slot = typename aligned_storage<sizeof(T), alignof(T)>::type;  
    struct Block
    {
        uint32_t block_size_;
        Slot* data_;
        vector<bool> used_;
        Block(uint32_t n):block_size_(n), data_(new Slot[n]), used_(n, false){}
        ~Block(){ delete[] data_; block_size_ = 0; used_.clear(); }
    };
    
    uint32_t block_size_;
    vector<T*> free_list_;
    vector<Block*> blocks_;

public:
    explicit GrowObjectMemory(uint32_t block_size):block_size_(block_size), free_list_(), blocks_(){}
    ~GrowObjectMemory()
    {
        block_size_ = 0;
        for (size_t i = 0; i < blocks_.size(); i++)
        {
            for (size_t j = 0; j < blocks_[i]->block_size_; j++)
            {
                if(blocks_[i]->used_[j] == true)
                {
                    T* ptr = reinterpret_cast<T*>(blocks_[i]->data_ + j);
                    ptr->~T();
                }
            }
            delete blocks_[i];
        }
        blocks_.clear();
        free_list_.clear();
    }

    template<typename... Args>
    T* allocate(Args&&... args)
    {
        if(free_list_.empty()) expand_list();

        T* ptr = free_list_.back();
        free_list_.pop_back();

        Block* m_block = index_of_blocks(ptr);
        
        uint32_t idx = static_cast<uint32_t>(reinterpret_cast<Slot*>(ptr) - m_block->data_);
        
        new (ptr) T(forward<Args>(args)...);
        m_block->used_[idx] = true;
        return ptr;
    }

    void deallocate(T* ptr)
    {
        Block* m_block = index_of_blocks(ptr);
        if(m_block == nullptr) 
        {
            cout << "the memory is not in pool" << "\n";
            return;
        }

        uint32_t idx = static_cast<uint32_t>(reinterpret_cast<Slot*>(ptr) - m_block->data_);
        if(m_block->used_[idx] == false) throw "the memory has been duplicated!!!";

        ptr->~T();
        free_list_.push_back(ptr);
        m_block->used_[idx] = false;
    }

private:
    void expand_list()
    {
        Block* m_block = new Block(block_size_);
        blocks_.push_back(m_block);

        for (size_t i = 0; i < m_block->block_size_; i++)
        {
            free_list_.push_back(reinterpret_cast<T*>(m_block->data_ + i));
        }
    }

    Block* index_of_blocks(T* ptr)
    {
        Slot* m_slot = reinterpret_cast<Slot*>(ptr);
        for (size_t i = 0; i < blocks_.size(); i++)
        {
            if(m_slot>=blocks_[i]->data_ && m_slot<(blocks_[i]->data_+static_cast<ptrdiff_t>(blocks_[i]->block_size_)))
            {
                return blocks_[i];
            }
        }
        return nullptr;
    }

};

namespace STL
{
struct CustomQueue
{
    uint32_t head;
    uint32_t tail;
    uint32_t capatity;
    void** data;
};
CustomQueue* init_queue(uint32_t size); //初始化队列
CustomQueue* expand_queue(CustomQueue * m_queue);
void push_front_val(CustomQueue* m_queue, void* value);
void* pop_front_val(CustomQueue* m_queue);

void push_back_val(CustomQueue *m_queue, void *value);
void* pop_back_val(CustomQueue *m_queue);

void destory_queue(CustomQueue *m_queue);
uint32_t queue_size(CustomQueue* m_queue);

};
template<typename T>
class CustomQueue
{
    static_assert(!is_reference<T>::value, "the T type is not reference!!!");
    using Slot = typename aligned_storage<sizeof(T), alignof(T)>::type;
public:
    CustomQueue(uint32_t capacity):capacity_(capacity), head_(0), tail_(0)
    {
        buffers_.resize(capacity);
    }
    ~CustomQueue()
    {
        for(uint32_t i = 0; i < Size(); i++)
        {
            uint32_t idx = ((head_ + i) % capacity_);
            T* ptr = ptr_index(idx);
            ptr->~T();
        }
        head_ = 0;
        tail_ = 0;
        capacity_ = 0;
        buffers_.clear();
    }

    template<typename... Args>
    void push_back_val(Args&&... args)
    {
        uint32_t cap = ((tail_ + 1) % capacity_);
        if(cap == head_)
            resize_queue();
        
        T* ptr = ptr_index(tail_);

        new (ptr) T(forward<Args>(args)...);
        
        tail_ = cap;
    }
    template<typename... Args>
    void push_front_val(Args&&... args)
    {
        uint32_t cap = ((head_ - 1) % capacity_);
        if(cap == tail_)
            resize_queue();

        T* ptr = ptr_index(cap);
        new (ptr) T(forward<Args>(args)...);
        head_ = cap;
    }

    T* pop_back_val()
    {
        uint32_t cap = ((tail_ - 1) % capacity_);
        T* ptr = ptr_index(cap);
        tail_ = cap;
        return ptr;
    }

    T* pop_front_val()
    {
        T* ptr = ptr_index(head_);
        head_ = ((head_ + 1) % capacity_);
        return ptr;
    }

    uint32_t Size()
    {
        return (tail_ - head_ + capacity_) % capacity_;
    }
private:
    T* ptr_index(uint32_t idx)
    {
        return reinterpret_cast<T*>(&buffers_[idx]);
    }
 
    const T* ptr_index(uint32_t idx) const
    {
        return reinterpret_cast<T*>(&buffers_[idx]);
    }
    void resize_queue()
    {
        vector<Slot> refresh_buffer;
        refresh_buffer.resize(capacity_ << 1);
        for (size_t i = 0; i < capacity_; i++)
        {
            T* old_ptr = ptr_index((i + head_) % capacity_);
            T* ptr = reinterpret_cast<T*>(&refresh_buffer[i]);
            
            if(is_lvalue_reference<T>::value)
            {
                new (ptr) T(*old_ptr);
            }
            else
            {
                new (ptr) T(move(*old_ptr));
            }
            old_ptr->~T();
        }
        capacity_ <<= 1;
        buffers_.swap(refresh_buffer);
    }

private:
    uint32_t capacity_;
    uint32_t head_;
    uint32_t tail_;
    vector<Slot> buffers_;
};

template<typename T>
class MinFunction;

template<typename Ret, typename... Args> 
class MinFunction<Ret(Args...)>
{
private:
    using Slot = typename aligned_storage<sizeof(void*)*3, alignof(void*)>::type;

    struct BaseCallBack
    {
        ~BaseCallBack() = default;
        virtual Ret invoke(Args... args) = 0;
        virtual BaseCallBack* clone_heap()  = 0;
        virtual BaseCallBack* clone_to_SBO(void* ptr, uint32_t size) = 0;
    };

    template<typename T>
    struct DrivideCallBack : public BaseCallBack
    {
        using FunType = typename decay<T>::type;
        FunType fun_;
        
        template<typename F>
        DrivideCallBack(F&& f):fun_(forward<F>(f)){}
        Ret invoke(Args... args) override
        {
            if constexpr (is_same<Ret, void>::value)
                fun_(forward<Args>(args)...);
            else
                return fun_(forward<Args>(args)...);
        }

        BaseCallBack* clone_heap() override
        {
            return (new DrivideCallBack(fun_));
        }
        BaseCallBack* clone_to_SBO(void* ptr, uint32_t size) override
        {
            if(sizeof(DrivideCallBack) <= size)
                return new (ptr) DrivideCallBack(fun_);
            return nullptr;
        }
    };

    BaseCallBack* callback_;
    Slot slot_;

    bool is_sbo()
    {
        void* e = reinterpret_cast<void*>(&slot_);
        void* s = reinterpret_cast<void*>(reinterpret_cast<void*>(&slot_) + sizeof(slot_));
        void* p = reinterpret_cast<void*>(callback_);
        return p>=e && p < s;
    }
    void clone_from(const MinFunction& other)
    {
        if(other.callback_ == nullptr) {delete callback_; callback_ = nullptr; return; }
        BaseCallBack* ptr = other.callback_->clone_to_SBO(&slot_, sizeof(slot_));
        if(ptr)
            callback_ = other.callback_;
        else
            callback_ = other.callback_->clone_heap();
    }

    void swap(const MinFunction& other)
    {
        if(!is_sbo() && !other.is_sbo())
        {
            swap(callback_, other.callback_);
            return;
        }
        MinFunction A(move(*this));
        MinFunction B(move(other));

        if(A.callback_)
        {
            if(A.is_sbo())
            {
                BaseCallBack* ptr = A.callback_->clone_to_SBO(&other.slot_, sizeof(other.slot_));
                if(ptr)
                    other.callback_ = ptr;
                else
                    other.callback_ = A.callback_->clone_heap();
                A.destory_current();
            }
            else
            {
                other.callback_ = A.callback_;
                A.callback_ = nullptr;
            }
        }
        else
            other.callback_ = nullptr;

        if(B.callback_)
        {
            if(B.is_sbo())
            {
                BaseCallBack* ptr = B.callback_->clone_to_SBO(&slot_, sizeof(slot_));
                if(ptr)
                    callback_ = ptr;
                else
                    callback_ = B.callback_->clone_heap();
                B.destory_current();
            }
            else
            {
                callback_ = B.callback_;
                B.callback_ = nullptr;
            }
        }
        else
            callback_ = nullptr;
    }

    void destory_current()
    {
        if(is_sbo())
        {
            callback_->~BaseCallBack();
            callback_ = nullptr;
        }
        else
        {
            delete callback_;
            callback_ = nullptr;
        }
    }

public:
    MinFunction():callback_(nullptr){}
    MinFunction(const MinFunction& other)
    {
        clone_from(other);
    }

    MinFunction& operator=(const MinFunction& other)
    {
        if(this != &other)
        {
            MinFunction temp(other);
            swap(temp);
        }
        return *this;
    }
    
    MinFunction(MinFunction&& other)
    {
        if(other.callback_ == nullptr) return;
        if(other.is_sbo())
        {
            BaseCallBack* ptr = other.callback_->clone_to_SBO(&slot_, sizeof(slot_));
            if(ptr)
                callback_ = ptr;
            else
                callback_ = other.callback_->clone_heap();
            
            delete other.callback_;
            other.callback_ = nullptr;
        }
        else
        {
            callback_ = other.callback_;
            other.callback_ = nullptr;
        }   
    }
    MinFunction& operator=(MinFunction&& other)
    {
        if(other != *this)
        {
            destory_current();
            if(other.is_sbo())
            {
                BaseCallBack* ptr = other.callback_->clone_to_SBO(&slot_, sizeof(slot_));
                if(ptr)
                    callback_ = ptr;
                else
                    callback_ = other.callback_->clone_heap();

                other.destory_current();
            }
            else
            {
                callback_ = other.callback_;
                other.callback_ = nullptr;
            }
        }
        return *this;
    }

    template<
        typename T,
        typename Decal = typename decay<T>::type,
        typename = typename enable_if<!is_same<Decal, MinFunction>::value>::type 
    >
    MinFunction(T&& f)
    {
        using FunType = DrivideCallBack<T>;
        if(sizeof(FunType) <= sizeof(slot_))
        {
            callback_ = new (&slot_) FunType(forward<T>(f));
        }
        else
            callback_ = new FunType(forward<T>(f));
    }
    
    template<typename T,
        typename Decal = typename decay<T>::type,
        typename = typename enable_if<!is_same<Decal, MinFunction>::value>::type 
    >
    MinFunction& operator=(T&& f)
    {
        MinFunction temp(forward<T>(f));
        swap(temp);
        return *this;
    }
    Ret operator()(Args... args)
    {
        return callback_->invoke(forward<Args>(args)...);
    }
};

template<typename T, uint32_t block_count = 256>
class ThreadPoolMemory
{
private:
    using Slot = typename aligned_storage<sizeof(T), alignof(T)>::type;
    static const uint32_t local_block_count = 32;
    static const uint32_t local_max_count   = 64;
    
    struct Node
    {
        Node* next;
        Slot storage_;
    };
    Node* node_from_ptr(uint8_t* ptr)
    {
        return reinterpret_cast<Node*>(ptr - offsetof(Node, storage_));
    }

    uint8_t* storage_from_node(Node* ptr)
    {
        return reinterpret_cast<uint8_t*>(&ptr->storage_);
    }

    struct GlobalMemory
    {
        mutex mtx;
        vector<uint8_t*> chunk_;
        atomic<uint64_t> used_counts_;
        Node* free_list_;
        GlobalMemory():used_counts_(block_count), free_list_(nullptr){}

        ~GlobalMemory() 
        {
            for (int i = 0; i < chunk_.size(); i++)
            {
                free(chunk_[i]);
            }
            used_counts_ = 0;
        }

        void extand_memory()
        {
            uint32_t off_set = offsetof(Node, storage_);
            uint32_t blockSIze = off_set + sizeof( typename aligned_storage<sizeof(T), alignof(T)>::type);
            uint64_t bytes = blockSIze*block_count;
            uint8_t* buff = (uint8_t*)malloc(bytes);

            chunk_.push_back(buff);

            for (int i = 0; i < (int)block_count; i++)
            {
                Node* curr = reinterpret_cast<Node*>(buff + i*blockSIze);
                curr->next = free_list_;
                free_list_ = curr;
                used_counts_.fetch_sub(1, memory_order_relaxed);
            }
        }

        Node* pop_memory(uint32_t count_size, uint32_t& pop_count)
        {
            lock_guard<mutex> lck(mtx);
            if(!free_list_) extand_memory();
            
            Node* curr = free_list_;

            Node* head = curr;
            uint32_t idx = 1;
            while (idx < count_size && curr->next)
            {
                curr = curr->next;
                idx++;
            }
            free_list_ = curr->next;
            curr->next = nullptr;

            pop_count = idx;
            return head;
        }

        void push_memory(Node* mem, uint32_t count_size)
        {
            if(mem == nullptr) return;
            lock_guard<mutex> lk(mtx);
            Node* head = mem;

            Node* tail = mem;
            uint32_t idx = 0;
            while (tail->next)
            {
                tail = tail->next;
                idx++;
            }
            tail->next = free_list_;
            free_list_ = head;

            used_counts_.fetch_add(idx, memory_order_relaxed);
        }
    };

    static GlobalMemory& Global_Instance()
    {
        static GlobalMemory global_mem;
        return global_mem;
    }

    struct LocalMemory
    {
        Node* free_list_ = nullptr;
        uint32_t free_count = 0;
        bool local_memory_empty()
        {
            return (free_count == 0);
        }

        ~LocalMemory()
        {
            if(free_list_)
            {   
                uint32_t idx = 1;
                Node* curr = free_list_;
                while (curr->next)
                {
                    curr = curr->next;
                    idx++;
                }
                ThreadPoolMemory::Global_Instance().push_memory(free_list_, idx);
                free_list_ = nullptr;
                free_count = 0;
            }
        }
    };

    static LocalMemory& Local_Instance()
    {
        static thread_local LocalMemory cache;
        return cache;
    }

    Node* pop_local_mem()
    {
        LocalMemory& local_mem = Local_Instance();
        if(local_mem.free_list_)
        {
            Node* curr = local_mem.free_list_;

            local_mem.free_list_ = curr->next;
            return curr;
        }
        uint32_t pop_count = 0;
        Node* mem = Global_Instance().pop_memory(local_block_count, pop_count);
        if(mem != nullptr && pop_count > 0)
        {
            local_mem.free_list_ = mem;
            Node* curr = local_mem.free_list_;
            local_mem.free_list_ = curr->next;
            curr->next = nullptr;
            // calculate rest
            Node* rest = local_mem.free_list_;
            while (rest)
            {
                rest = rest->next;   
                local_mem.free_count++;
            }
            return curr;
        }
    }

    void push_local_mem(Node* ptr)
    {
        if(ptr == nullptr) return;

        LocalMemory& local_mem = ThreadPoolMemory::Local_Instance();
        ptr->next = local_mem.free_list_;
        local_mem.free_list_ = ptr;
        local_mem.free_count++;

        if(local_mem.free_count >= local_max_count)
        {
            uint32_t idx = 1;
            Node* curr = local_mem.free_list_;
            while (curr->next && idx <= local_block_count)
            {
                idx++;
                curr = curr->next;
            }
            Node* head = curr->next;
            curr->next = nullptr;

            ThreadPoolMemory::Global_Instance().push_memory(local_mem.free_list_, idx);
            local_mem.free_list_ = nullptr;

            local_mem.free_count -= idx;
            local_mem.free_list_ = head;
        }
    }

public:
    template<typename... Args>
    T* allocate(Args... args)
    {
        Node* m_node = pop_local_mem();
        uint8_t* mem = storage_from_node(m_node);
        
        T* ptr = new (mem) T(forward<Args>(args)...);

        return ptr;
    }

    void deallocate(T* ptr)
    {
        if(ptr == nullptr) return;
        ptr->~T();
        push_local_mem(node_from_ptr(reinterpret_cast<uint8_t*>(ptr)));
    }
};

enum class Color { RED, BLACK };

template<typename T, typename Comp = std::less<T>>
class CustomSet
{
public:
    struct CustomTreeNode
    {
        T value;
        Color color;
        CustomTreeNode* parent;
        CustomTreeNode* left;
        CustomTreeNode* right;

        CustomTreeNode(T val) : value(val), color(Color::BLACK), parent(nullptr), left(nullptr), right(nullptr){}
        CustomTreeNode(): value(T()), color(Color::BLACK), parent(nullptr), left(nullptr), right(nullptr){}
    
    };

    CustomSet()
    {
        nil = new CustomTreeNode();
        nil->left = nil;
        nil->right = nil;
        nil->parent = nil;
        nil->color = Color::BLACK;
        root = nil;
    }
    ~CustomSet()
    {
        clear();
        delete nil;
    }
    class Iterator
    {
    using Iter_Reference = T&;
    using Iter_ValueType = T;
    using Iter_Pointer   = T*;

    private:
        CustomTreeNode* curr_;
        CustomTreeNode* nil_;
        CustomTreeNode* root_;
    public:
        Iter_Reference operator*() const { return curr_->value; }
        Iter_Pointer   operator->() const { return &curr_->value; }
        Iterator(CustomTreeNode* curr, CustomTreeNode* nil, CustomTreeNode* root):
            curr_(curr), nil_(nil), root_(root){}

        Iterator& operator++()
        {
            curr_ = next(curr_);
            return *this;
        }

        Iterator operator++(int)
        {
            Iterator temp = *this;
            ++(*this);
            return temp;
        }
        Iterator& operator--()
        {
            curr_ = prev(curr_);
            return *this;
        }
        Iterator operator--(int)
        {
            Iterator temp = *this;
            --(*this);
            return temp;
        }

        bool operator==(const Iterator& other) const { return curr_ == other.curr_; }
        bool operator!=(const Iterator& other) const { return curr_ != other.curr_; }
    private:
        CustomTreeNode* next(CustomTreeNode* node)
        {
            if(node == nil_) return nil_;
            if(node->right != nil_)
            {
                CustomTreeNode* large_node = node->right;
                while (large_node->left != nil_)
                    large_node = large_node->left;
                
                return large_node;
            }

            CustomTreeNode* large_node = node->parent;
            while (large_node!=nil_ && node == large_node->right)
            {
                node = large_node;
                large_node = large_node->parent;
            }
            return large_node;
        }

        CustomTreeNode* prev(CustomTreeNode* node)
        {
            if(node == nil_) return nil_;
            if(node->left != nil_)
            {
                CustomTreeNode* large_node = node->left;
                while (large_node->right != nil_)
                    large_node = large_node->right;
                
                return large_node;
            }

            CustomTreeNode* large_node = node->parent;
            while (large_node!=nil_ && node == large_node->left)
            {
                node = large_node;
                large_node = large_node->parent;
            }
            return large_node;
        }

    friend CustomSet;
    };
    void insert(const T& val)
    {
        CustomTreeNode* node = new CustomTreeNode(val);
        CustomTreeNode* curr = root, *ptr = nil;
        while (curr != nil)
        {
            ptr = curr;
            if(Comp()(curr->value, node->value))
                curr = curr->right;
            else if(Comp()(node->value, curr->value))
                curr = curr->left;
            else
            {
                delete node;
                return;
            }
        }
        if(ptr == nil)
            root = node;
        else if(Comp()(node->value, ptr->value))
            ptr->left = node;
        else
            ptr->right = node;

        node->parent = ptr;
        node->left = nil;
        node->right = nil;
        node->color = Color::RED;

        insert_fix(node);
    }
    void erase(const T& value)
    {
        CustomTreeNode* curr = root;
        while (curr != nil)
        {
            if(Comp()(curr->value, value))
                curr = curr->right;
            else if(Comp()(value, curr->value))
                curr = curr->left;
            else
                break;
        }
        if(curr != nil)
            erase_node(curr);
    }

private:
    void left_rotate(CustomTreeNode* node)
    {
        CustomTreeNode* right = node->right;
        assert(right != nullptr);
        node->right = right->left;
        if(right->left != nil) right->left->parent = node;

        right->parent = node->parent;
        if(node->parent == nil)
            root = right;
        else if(node == node->parent->left)
            node->parent->left = right;
        else
            node->parent->right = right;

        right->left = node;
        node->parent = right;
    }
    void right_rotate(CustomTreeNode* node)
    {
        CustomTreeNode* left = node->left;
        assert(left != nullptr);
        node->left = left->right;
        if(left->right != nil) left->right->parent = node;

        left->parent = node->parent;
        if(node->parent == nil)
            root = left;
        else if(node == node->parent->left)
            node->parent->left = left;
        else
            node->parent->right = left;

        left->right = node;
        node->parent = left;
    }
    void insert_fix(CustomTreeNode* node)
    {
        CustomTreeNode* curr = node;
        while (curr->parent->color == Color::RED)
        {
            if(curr->parent == curr->parent->parent->left)
            {
                CustomTreeNode* uncle = curr->parent->parent->right;
                if(uncle->color == Color::RED)
                {
                    uncle->color = Color::BLACK;
                    curr->parent->color = Color::BLACK;
                    curr->parent->parent->color = Color::RED;
                    curr = curr->parent->parent;
                }
                else
                {
                    if(curr == curr->parent->right)
                    {
                        curr = curr->parent;
                        left_rotate(curr);
                    }
                    curr->parent->color = Color::BLACK;
                    curr->parent->parent->color = Color::RED;
                    right_rotate(curr->parent->parent);
                }
            }
            else
            {
                CustomTreeNode* uncle = curr->parent->parent->left;
                if(uncle->color == Color::RED)
                {
                    uncle->color = Color::BLACK;
                    curr->parent->color = Color::BLACK;
                    curr->parent->parent->color = Color::RED;
                    curr = curr->parent->parent;
                }
                else
                {
                    if(curr == curr->parent->left)
                    {
                        curr = curr->parent;
                        right_rotate(curr);
                    }
                    curr->parent->color = Color::BLACK;
                    curr->parent->parent->color = Color::RED;
                    left_rotate(curr->parent->parent);
                }
            }
            if(curr == root) break;
        }
        root->color = Color::BLACK;
    }

    void swap_two_node(CustomTreeNode* orig, CustomTreeNode* alter)
    {
        if(orig->parent == nil)
            root = alter;
        else if(orig == orig->parent->left)
            orig->parent->left = alter;
        else
            orig->parent->right = alter;

        alter->parent = orig->parent;
    }
    CustomTreeNode* traverse_tiny_node(CustomTreeNode* node)
    {
        while (node->left != nil) node = node->left;
        return node;
    }
    void erase_fix(CustomTreeNode* node)
    {
        while (node != root && node->color == Color::BLACK)
        {
            if(node == node->parent->left)
            {
                CustomTreeNode* bro = node->parent->right;
                if(bro->color == Color::RED)
                {
                    bro->color = Color::BLACK;
                    bro->parent->color = Color::RED;

                    left_rotate(node->parent);

                    bro = node->parent->right;
                }

                if(bro->left->color == Color::BLACK && bro->right->color == Color::BLACK)
                {
                    bro->color = Color::RED;
                    node = node->parent;
                }
                else
                {
                    if(bro->right->color == Color::BLACK)
                    {
                        bro->left->color = Color::BLACK;
                        bro->color = Color::RED;
                        right_rotate(bro);

                        bro = node->parent->right;
                    }
                    bro->color = node->parent->color;
                    node->parent->color = Color::BLACK;
                    bro->right->color = Color::BLACK;

                    left_rotate(node->parent);
                    node = root;
                }
            }
            else
            {
                CustomTreeNode* bro = node->parent->left;
                if(bro->color == Color::RED)
                {
                    bro->color = Color::BLACK;
                    bro->parent->color = Color::RED;

                    right_rotate(node->parent);

                    bro = node->parent->left;
                }

                if(bro->left->color == Color::BLACK && bro->right->color == Color::BLACK)
                {
                    bro->color = Color::RED;
                    node = node->parent;
                }
                else
                {
                    if(bro->left->color == Color::BLACK)
                    {
                        bro->right->color = Color::BLACK;
                        bro->color = Color::RED;
                        left_rotate(bro);

                        bro = node->parent->left;
                    }
                    bro->color = node->parent->color;
                    node->parent->color = Color::BLACK;
                    bro->left->color = Color::BLACK;

                    right_rotate(node->parent);
                    node = root;
                }
            }
        }
        node->color = Color::BLACK;
    }
    void erase_node(CustomTreeNode* node)
    {
        CustomTreeNode* curr = nil, *del_node = node;
        Color color = del_node->color;
        if(node->left == nil)
        {
            curr = node->right;
            swap_two_node(node, node->right);
        }
        else if(node->right == nil)
        {
            curr = node->left;
            swap_two_node(node, node->left);
        }
        else
        {
            del_node = traverse_tiny_node(node->right);
            curr = del_node->right;

            color = del_node->color;
            if(del_node->parent == node)
            {
                curr->parent = del_node;
            }
            else
            {
                swap_two_node(del_node, del_node->right);
                del_node->right = node->right;
                del_node->right->parent = del_node;
            }
            swap_two_node(node, del_node);
            del_node->left = node->left;
            del_node->left->parent = del_node;

            del_node->color = node->color;
        }

        if(color == Color::BLACK)
            erase_fix(curr);
        
        delete node;
    }
    void clear()
    {
        if(root == nil) return;

        CustomTreeNode* curr = root;
        CustomTreeNode* last_visited = nil;

        stack<CustomTreeNode*> nodes;
        while (curr != nil || !nodes.empty())
        {
            if(curr != nil)
            {
                nodes.push(curr);
                curr = curr->left;
            }
            else
            {
                CustomTreeNode* top = nodes.top();
                if(top->right != nil && top->right != last_visited)
                {
                    curr = top->right;
                }
                else
                {
                    delete top;
                    last_visited = top;
                    nodes.pop();
                }
            }
        }
        root = nil;
    }
    CustomTreeNode* min_val_node(CustomTreeNode* node)
    {
        while (node->left != nil)
            node = node->left;
        return node;
    }
    CustomTreeNode* max_val_node(CustomTreeNode* node)
    {
        while (node->right != nil)
            node = node->right;
        return node;
    }
public:
    Iterator begin()
    {
        if(root == nil) return Iterator(nil, nil, root);
        
        CustomTreeNode* min_node = min_val_node(root);

        return Iterator(min_node, nil, root);
    }

    Iterator end()
    {
        return Iterator(nil, nil, root);
    }

private:
    CustomTreeNode* root;
    CustomTreeNode* nil;
};

struct PortSignal
{
    struct Dim
    {
        int lsb;
        int msb;
        Dim* dim;
        Dim():lsb(0), msb(0), dim(nullptr){}
    };
    string name;
    Dim dim;
    PortSignal():name(""),dim(Dim{}){}
};

template<typename T, typename Compare = less<T> >
class CustMultiSet
{
private:
    struct CustomTreeNode
    {
        T value_;
        Color color_;
        CustomTreeNode* parent_;
        unique_ptr<CustomTreeNode> left_;
        unique_ptr<CustomTreeNode> right_;

        CustomTreeNode(const T& value):value_(value), color_(Color::BLACK), parent_(nullptr), left_(), right_(){}
    };
    unique_ptr<CustomTreeNode> root_;
    uint32_t size = 0;
public:
    void insert(const T& value)
    {
        CustomTreeNode* curr = root_.get();
        CustomTreeNode* ptr = nullptr;
        while (curr != nullptr)
        {
            ptr = curr;
            if(Compare()(value, curr->value_))
                curr = curr->left_.get();
            else
                curr = curr->right_.get();
        }
        unique_ptr<CustomTreeNode> node(new CustomTreeNode(value));
        node->color_ = Color::RED;

        CustomTreeNode* fixed_ptr = nullptr;
        if(root_ == nullptr)
        {
            root_ = move(node);
            fixed_ptr = root_.get();
        }
        else if(Compare()(value, ptr->value_))
        {
            ptr->left_ = move(node);
            ptr->left_->parent_ = ptr;
            fixed_ptr = ptr->left_.get();
        }
        else
        {
            ptr->right_ = move(node);
            ptr->right_->parent_ = ptr;
            fixed_ptr = ptr->right_.get();
        }
        insert_fix(fixed_ptr);
        size++;
    }

    void erase(const T& value)
    {
        CustomTreeNode* curr = root_.get();
        CustomTreeNode* ptr = nullptr;
        while (curr != nullptr)
        {
            ptr = curr;
            if(Compare()(curr->value_, value))
                curr = curr->right_.get();
            else if(Compare()(value, curr->value_))
                curr = curr->left_.get();
            else
                break;
        }
        
        if(ptr == nullptr) return;
        erase_node(ptr);
        size--;
    }

private:
    Color color_of(CustomTreeNode* node)
    {
        return node == nullptr? Color::BLACK:node->color_;
    }
    unique_ptr<CustomTreeNode>& get_parent_node(CustomTreeNode* node)
    {
        if(node->parent_ == nullptr)
            return root_;
        
        if(node == node->parent_->left_.get())
            return node->parent_->left_;
        else
            return node->parent_->right_;
    }

    void left_rotate(CustomTreeNode* node)
    {
        CustomTreeNode* parent = node->parent_;
        unique_ptr<CustomTreeNode>& x_link = get_parent_node(node);

        unique_ptr<CustomTreeNode> x_up = move(x_link);
        CustomTreeNode* x_uptr = x_up.get();

        unique_ptr<CustomTreeNode> y_up = move(x_up->right_);
        CustomTreeNode* y_uptr = y_up.get();

        x_up->right_ = move(y_up->left_);
        if(x_up->right_ != nullptr) x_up->right_->parent_ = x_uptr;

        y_up->left_ = move(x_up);
        y_up->left_->parent_ = y_uptr;

        y_up->parent_ = parent;

        x_link = move(y_up);
    }
    void right_rotate(CustomTreeNode* node)
    {
        CustomTreeNode* parent = node->parent_;
        unique_ptr<CustomTreeNode>& x_link = get_parent_node(node);

        unique_ptr<CustomTreeNode> x_up = move(x_link);
        CustomTreeNode* x_uptr = x_up.get();

        unique_ptr<CustomTreeNode> y_up = move(x_up->left_);
        CustomTreeNode* y_uptr = y_up.get();

        x_up->left_ = move(y_up->right_);
        if(x_up->left_ != nullptr) x_up->left_->parent_ = x_uptr;

        y_up->right_ = move(x_up);
        y_up->right_->parent_ = y_uptr;

        y_up->parent_ = parent;

        x_link = move(y_up);
    }

    void insert_fix(CustomTreeNode* node)
    {
        while (node != root_.get() && color_of(node->parent_) == Color::RED)
        {
            if(node->parent_ == nullptr) break;
            if(node->parent_ == node->parent_->parent_->left_.get())
            {
                CustomTreeNode* uncle = node->parent_->parent_->right_.get();
                if(color_of(uncle) == Color::RED)
                {
                    uncle->color_ = Color::BLACK;
                    node->parent_->color_ = Color::BLACK;
                    node->parent_->parent_->color_ = Color::RED;
                    node = node->parent_->parent_;
                }
                else
                {
                    if(node == node->parent_->right_.get())
                    {
                        node = node->parent_;
                        left_rotate(node);
                    }

                    node->parent_->color_ = Color::BLACK;
                    node->parent_->parent_->color_ = Color::RED;
                    right_rotate(node->parent_->parent_);
                }
            }
            else
            {
                CustomTreeNode* uncle = node->parent_->parent_->left_.get();
                if(color_of(uncle) == Color::RED)
                {
                    uncle->color_ = Color::BLACK;
                    node->parent_->color_ = Color::BLACK;
                    node->parent_->parent_->color_ = Color::RED;
                    node = node->parent_->parent_;
                }
                else
                {
                    if(node == node->parent_->left_.get())
                    {
                        node = node->parent_;
                        right_rotate(node);
                    }

                    node->parent_->color_ = Color::BLACK;
                    node->parent_->parent_->color_ = Color::RED;
                    left_rotate(node->parent_->parent_);
                }
            }
        }
        root_->color_ = Color::BLACK;
    }

    void erase_node(CustomTreeNode* node)
    {
        auto min_val_node = [](CustomTreeNode* x) -> CustomTreeNode*
        {
            while(x->left_ != nullptr) x = x->left_;
            return x;
        };

        Color color = color_of(node);
        CustomTreeNode* curr = nullptr, *y_ptr = node;

        unique_ptr<CustomTreeNode> y_up;
        if(node->right_ == nullptr)
        {
            unique_ptr<CustomTreeNode>& up_node = get_parent_node(node);
            y_up = move(up_node);

            curr = y_up->left_.get();

            up_node = move(y_up->left_);

            if(up_node != nullptr) up_node->left_->parent_ = y_up.get();
        }
        else if(node->left_ == nullptr)
        {
            unique_ptr<CustomTreeNode>& up_node = get_parent_node(node);
            y_up = move(up_node);

            curr = y_up->right_.get();

            up_node = move(y_up->right_);

            if(up_node != nullptr) up_node->right_->parent_ = y_up.get();
        }
        else
        {
            curr = min_val_node(node->right_.get());
            // y_ptr = curr->right_

            


        }

        if(color == Color::BLACK)
            erase_fix(curr);
    }

    void erase_fix(CustomTreeNode* node)
    {
        

    }

};

class Screen
{
public:
    Screen() = default;
    Screen(int x):x(x) { cout << "this pointer: " << this << ", " << "x: " << x << "\n"; }
    static void* operator new(size_t m_size);
    static void operator delete(void* ptr) noexcept;

private:
    struct StoreNode
    {
        StoreNode* next = nullptr;
    };

    static StoreNode* freeStore;
    static const int maxStore;
    int x;
};

class AirPlane
{
private:
    struct Require
    {
        string name_;
        int id_;
        Require() = delete;
        Require(const string& name, int id):name_(name), id_(id){}
        ~Require() = default;
    };

    union StoreUnion
    {
        Require req_;
        AirPlane* next;
        StoreUnion() {};
        ~StoreUnion() {};
    } store_;

    enum class State { FREE, USED };
    State state_;

    static AirPlane* freeStore;
    static const int maxCount;
public:
    AirPlane() = delete;
    AirPlane(const string& name, int id):state_(State::USED)
    {
        new (&store_.req_) Require(name, id);
        cout << "this: " << this << ". Requires name: " << store_.req_.name_ << " and id: "<< store_.req_.id_ << "\n";
    }
    ~AirPlane()
    {
        if(state_ == State::USED)
        {
            store_.req_.~Require();
        }
    }

    static void* operator new(size_t m_size);
    static void operator delete(void* mem) noexcept;
};

vector<string> wordBreak1(string s, vector<string>& wordDict);
bool hasCycle(ListNode *head);

struct VideoInfo
{
    string name;
    string path;
    double size; //g
    VideoInfo() {}
    VideoInfo(string name, string path, double size):name(name), path(path),size(size) {}
};


// vector<VideoInfo> traverse_file_on_direction(const string& src_dir, const string& suffix);
// void transform_quality(const vector<VideoInfo>& files, const string& dst_dir);
// void copy_file(const string& src_path, const string& dst_path);