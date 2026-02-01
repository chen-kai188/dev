#include "LeetCode.h"
// #include <opencv4/opencv2/core.hpp>
// #include <opencv4/opencv2/opencv.hpp>

ListNode *reverseKGroup(ListNode *head, int k)
{
    ListNode* swap_list = new ListNode();
    swap_list->next = head;

    ListNode* ptr_front = swap_list;
    ListNode* ptr_back = swap_list;

    while (ptr_back)
    {
        /* code */
        for(int i = 0; (i < k)&&(ptr_back); i++) ptr_back = ptr_back->next;
        if(ptr_back == nullptr) break;
        
        ListNode* start_group = ptr_front->next;
        ListNode* next_group = ptr_back->next;  
        ptr_back->next = nullptr;

        ptr_front->next = reverseListNode(start_group);
        
        start_group->next = next_group;
        ptr_front       = start_group;
        ptr_back        = start_group;
    }
    return swap_list->next;
}

ListNode *reverseListNode(ListNode *head)
{
    ListNode* new_ListNode = nullptr;
    ListNode* cur = head;
    while (cur)
    {
        /* code */
        ListNode* nextNode = cur->next;
        cur->next = new_ListNode;
        new_ListNode = cur;
        cur = nextNode;
    }

    return new_ListNode;
}

ListNode *createListNode(initializer_list<int> arr)
{
    ListNode* list_node = new ListNode();
    list_node->next = nullptr;
    
    ListNode* ptr_list = list_node;
    for (auto iter = arr.begin(); iter != arr.end(); iter++)
    {
        /* code */
        ListNode* temp = new ListNode(*iter);
        ptr_list->next = temp;
        ptr_list = temp;
    }   

    return list_node->next;
}

int removeElement(vector<int> &nums, int val)
{
    if(nums.size() == 0)
        return 0;

    int pos_num = 0, index_num = 0;
    while (index_num < nums.size())
    {
        /* code */
        if(nums[index_num] != val)
        {
            nums[pos_num++] = nums[index_num];
        }
        index_num++;
    }
    return pos_num;
}

int strStr(string haystack, string needle)
{
    // Naive Solution
    // int haystack_size = haystack.size(), needle_size = needle.size();
    // if(haystack_size < needle_size)
    //     return -1;

    // for (size_t i = 0; i < haystack_size; i++)
    // {
    //     /* code */
    //     for (size_t j = 0; (j < needle_size) && ((i + j) < haystack_size); j++)
    //     {
    //         /* code */
    //         if(needle.at(j) != haystack.at(j + i))
    //             break;
    //         else if((j + 1) == needle_size)
    //             return i;
    //     }
    // }
    // return -1;
    
    //get next array
    int needle_size = needle.size();
    vector<int> next_pi(needle_size);

    int j = 0, k = 0;
    for(int i = 1; i < needle_size; i++)
    {
        while(j > 0 && needle[i] != needle[j])
            j = next_pi[j - 1];
        
        if(needle[i] == needle[j])
            j++;
        
        next_pi[i] = j;
    }

    int haystack_size = haystack.size();
    k = 0, j = 0;
    for (size_t k = 0; k < haystack_size; k++)
    {
        while( j > 0 && haystack[k] != needle[j])
            j = next_pi[j - 1];
        
        if(haystack[k] == needle[j])
            j++;
        
        if( j == needle_size)
            return (k + 1 - j);
    }

    return -1;
}

vector<int> findSubstring(string s, vector<string> &words)
{   
    set<string> splice_str_vec;
    back_trace(0, words, splice_str_vec);

    vector<int> rets;
    for (auto iter = splice_str_vec.begin(); iter != splice_str_vec.end(); iter++)
    {
        int needle_size = iter->size();
        if(s.size() < needle_size)
            return rets;

        int j = 0;
        vector<int> needle(needle_size);
        for(int k = 1; k < needle_size; k++)
        {
            while( j > 0 && (*iter)[k] != (*iter)[j])
                j = needle[j - 1];
            if((*iter)[k] == (*iter)[j])
                j++;
            
            needle[k] = j;
        }

        j = 0;
        int source_size = s.size();
        for (size_t k = 0; k < source_size; k++)
        {
            /* code */
            while (j > 0 && s[k] != (*iter)[j])
            {
                /* code */
                j = needle[j - 1];
            }
            if(s[k] == (*iter)[j])
                j++;

            if(j == needle_size)
            {
                rets.push_back(k - j + 1);
            }
        }
    }
    return rets;
}

void back_trace(int curr_index, vector<string>& words, set<string>& rets)
{
    if(curr_index == words.size() - 1)
    {
        string splice_str;
        for (size_t i = 0; i < words.size(); i++)
        {
            /* code */
            splice_str.append(words[i]);
        }
        cout << splice_str << endl;
        rets.insert(splice_str);
        return;
    }    
    for(size_t i = curr_index; i < words.size(); i++)
    {  
        swap(words[i], words[curr_index]);
        back_trace(curr_index + 1, words, rets);
        swap(words[curr_index], words[i]);
    }
}

vector<int> findSubstring1(string s, vector<string> &words)
{
    vector<int> rets;

    int words_size = 0;
    map<string, int> words_statistics;
    for(int i = 0; i < words.size(); i++)
    {
        words_statistics[words[i]]++;
        words_size += words[i].size();
    }
    if(s.size() < words_size)
        return rets;

    map<string, int> s_statistics;
    int single_words_size = words[0].size();
    for (size_t i = 0; i <= (s.size() - words_size); i += 1)
    {
        /* code */
        int j = i;
        for(; j < words_size + i; j += single_words_size)
        {
            string partial_str = s.substr(j, single_words_size);
            if(words_statistics.find(partial_str) == words_statistics.end())
            {
                break;
            }          
            else
            {
                s_statistics[partial_str]++;
                if(s_statistics[partial_str] > words_statistics[partial_str])
                {
                    break;
                }
            }
        }
        if( j == i + words_size)
        {
            rets.push_back(i);
        }
        s_statistics.clear();
    }
    return rets;
}

void nextPermutation(vector<int> &nums)
{
    int size = static_cast<int>(nums.size());
    int i = size - 2;
    for(; i >= 0; i--)
    {
        if(nums[i] < nums[i + 1])
            break;
    }

    if(i < 0)
        return reverse(nums.begin(), nums.end());
    
    for (int j = size - 1; j >= i; j--)
    {
        /* code */
        if(nums[j] > nums[i])
        {
            swap(nums[j], nums[i]);
            break;
        }
    }
    reverse(nums.begin() + i + 1, nums.end());
}

int longestValidParentheses(string s)
{
    int size = s.size(), effective_nums = 0;
    stack<int> brackets; brackets.push(-1);

    for(int i = 0; i < size; i++)
    {
        if(s[i] == '(')
        {
            brackets.push(i);
        }
        else
        {
            brackets.pop();
            if(brackets.empty()) 
            {
                brackets.push(i);
            }
            else
            {
                effective_nums = max(effective_nums, i - brackets.top());
            } 
        }
    }
    return effective_nums;
}

int search(vector<int> &nums, int target)
{
    // int rets = -1;
    // int size = nums.size();
    // for (int i = 0; i < size; i++)
    // {
    //     /* code */
    //     if(nums[i] == target)
    //     {
    //         rets = i;
    //         break;
    //     }
    // }
    // return rets;

    int rets = -1;
    int right = nums.size() - 1, left = 0, mid = 0;
    while(left <= right)
    {
        mid = (left + right) >> 1;
        if(nums[mid] == target) 
        {
            rets = mid;
            break;
        }

        if(nums[left] <= nums[mid])
        {
            (nums[left] <= target && target <= nums[mid]) ? right = mid - 1 : left = mid + 1;
        }
        else
        {
            (target < nums[left] && target > nums[mid]) ? left = mid + 1 : right = mid - 1;
        }
    }
    return rets;
}

int maxProfit(vector<int> &prices)
{   
    int max_price = 0, min_price = 1e9;
    for(size_t i = 0; i < prices.size(); i++)
    {
        max_price = max(max_price, prices[i] - min_price);
        min_price = min(min_price, prices[i]);
    }
    return max_price;
}

vector<int> searchRange(vector<int> &nums, int target)
{
    vector<int> rets(2, -1);
    int right = nums.size() - 1, left = 0;
    while (left <= right)
    {
        /* code */
        if(nums[left] == target && nums[right] == target)
        {
            rets[0] = left;
            rets[1] = right;
            break;
        }

        if(nums[left] < target)
            left++;
        if(nums[right] > target)
            right--;

    }
    return rets;
}

int searchInsert(vector<int> &nums, int target)
{
    int ans = nums.size();
    int right = nums.size() - 1, left = 0, mid = 0;
    while (left <= right)
    {
        /* code */
        mid = (left + right) >> 1;
        if(nums[mid] >= target)
        {
            ans = mid;
            right = mid - 1;
        }
        else
            left = mid + 1;
    }
    
    return ans;
}

bool isValidSudoku(vector<vector<char> > &board)
{
    size_t rows = board.size();
    size_t cols = board.size();
    for (size_t i = 0; i < rows; i++)
    {
        /* code */
        for(size_t j = 0; j < cols; j++)
        {
            if((i + 1) % 3 == 0 && (j + 1) % 3 == 0)
            {
                if(!is_unique(board, i, j))
                    return false;
            }
            if(i == j)
            {
                set<char> check_row;
                for (size_t k = 0; k < rows; k++)
                {
                    if(board[i][k] == '.') continue;
                    if(check_row.count(board[i][k]))
                        return false;
                    check_row.insert(board[i][k]);
                }
                set<char> check_col;
                for (size_t k = 0; k < cols; k++)
                {
                    if(board[k][i] == '.') continue;
                    if(check_col.count(board[k][i]))
                        return false;
                    check_col.insert(board[k][i]);
                }
            }
        }
    }
    return true;
}
bool is_unique(vector<vector<char> > &board, int row, int col)
{
    set<char> check_unique;
    for(int i = row - 2; i <= row; i++)
    {
        for(int j = col - 2; j <= col; j++)
        {
            char C = board[i][j];
            if(C == '.') continue;
            if(check_unique.find(C) != check_unique.end())
                return false;
            check_unique.insert(C);
        }
    }
    return true;
}

void solveSudoku(vector<vector<char> > &board)
{
    int rows = board.size();
    int cols = board.size();

    vector<int> row_score(9), col_score(9);
    vector<vector<int> > blocks(rows/3, vector<int>(cols / 3));
    vector<pair<int, int> > scores;
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            if(board[i][j] == '.')
                scores.emplace_back(i, j);
            else
            {
                int digital = static_cast<int>(board[i][j] - '0' - 1);
                flip(i, j, digital, row_score, col_score, blocks);
            }
        }
    }
    bool is_vaild = false;
    back_trace_sudo(0, board, scores, row_score, col_score, blocks, is_vaild);
}
void flip(int i, int j, int digital, vector<int> &rows, vector<int> &cols, vector<vector<int>> &blocks)
{
    rows[i] ^= (1 << digital);
    cols[j] ^= (1 << digital);
    blocks[i / 3][j / 3] ^= (1 << digital);
}

void back_trace_sudo(int index, vector<vector<char> > &board, const vector<pair<int, int>> &score, vector<int> &rows, vector<int> &cols, vector<vector<int>> &blocks, bool& is_vaild)
{
    if(index == static_cast<int>(score.size()))
    {
        is_vaild = true;
        return;
    }
        
    
    int i = score[index].first;
    int j = score[index].second;

    int mode = ~(rows[i] | cols[j] | blocks[i / 3][j / 3]) & 0x1ff;
    for(; mode&&!is_vaild; mode &= (mode - 1))
    {
        int digital_mask = (mode & (-mode));
        int digital =  __builtin_ctz(digital_mask);
        flip(i, j, digital, rows, cols, blocks);
        board[i][j] = digital + '0' + 1;
        back_trace_sudo(index + 1, board, score, rows, cols, blocks, is_vaild);
        flip(i, j, digital, rows, cols, blocks);
    }
}

string countAndSay(int n)
{
    string convert_digital = "1";
    while (--n)
    {
        vector<pair<char, int> > digital_vec;
        for (size_t i = 0; i < convert_digital.size(); i++)
        {
            /* code */
            char C = convert_digital[i];

            int size = static_cast<int>(digital_vec.size());
            if(size == 0)
            {
                digital_vec.push_back(make_pair(C, 1));
                continue;
            }

            if(digital_vec[size - 1].first == C)
                digital_vec[size - 1].second++;
            else
                digital_vec.push_back(make_pair(C, 1));
        }

        convert_digital.clear(); //update later
        for(size_t i = 0; i < digital_vec.size(); i++)
        {
            convert_digital += to_string(digital_vec[i].second) + digital_vec[i].first;
        }
    }
    return convert_digital;
}

vector<vector<int>> combinationSum(vector<int> &candidates, int target)
{
    vector<int> ret;
    vector<vector<int>> ans;
    back_trace_combination(candidates, target, ret, ans);
    return ans;
}

void back_trace_combination(const vector<int> &candidates, int target, vector<int>& ret, vector<vector<int>> &ans, int index)
{
    int size = static_cast<int>(candidates.size());
    if(index == size || target < 0) return;
    if(target == 0)
    {
        ans.push_back(ret);
        return;
    }
    
    for (int i = index; i < size; i++)
    {
        /* code */
        ret.push_back(candidates[i]);
        back_trace_combination(candidates, target - candidates[i], ret, ans, i);
        ret.pop_back();
    }
}

vector<vector<int>> combinationSum2(vector<int>& candidates, int target)
{
    sort(candidates.begin(), candidates.end());
    vector<int> ret;
    vector<vector<int>> ans;
    back_trace_combination2(candidates, target, ret, ans);
    return vector<vector<int>>();
}

void back_trace_combination2(const vector<int> &candidates, int target, vector<int> &ret, vector<vector<int>> &ans, int index)
{
    if(index == (static_cast<int>(candidates.size()) + 1) || target < 0) return;
    if(target == 0)
    {
        ans.push_back(ret);
        return;
    }

    for (size_t i = index; i < candidates.size(); i++)
    {
        /* code */
        if(i > index && candidates[i] == candidates[i - 1]) continue;
        ret.push_back(candidates[i]);
        back_trace_combination2(candidates, target - candidates[i], ret, ans, i + 1);
        ret.pop_back();
    }

}

int firstMissingPositive(vector<int> &nums)
{
    sort(nums.begin(), nums.end());

    int target = 1;
    for(size_t i = 0; i < nums.size(); i++)
    {
        if(nums[i] <= 0 || (i > 0 && (nums[i] == nums[i - 1]))) continue;
        if(nums[i] != target) break;
        target++;
        // target = nums[i] + 1;
        // if((i > 0) && (nums[i] - nums[i - 1]) != 1) break;
    }
    return target;
}
int trap(vector<int> &height)
{
    long int max_area = 0;
    int max_val = *max_element(height.begin(), height.end());
    for (int i = 0; i < max_val; i++)
    {
        size_t pos = 0;
        for(size_t j = pos; j < height.size(); j++)
        {
            if(height[j] == 0) continue;
            height[j]--;

            pos = j + 1;
            while(pos < height.size())
            {
                if(height[pos] != 0)
                    break;
                pos++;
            }

            if(pos < height.size())
            {
                max_area += (pos - j - 1);
            }  
            j = pos - 1;
        }
        cout << "the height: "<< i << "; max_area= " << max_area << " \n";
    }
    return max_area;
}
int trap2(vector<int> &height)
{
    int max_area = 0;
    stack<int> st;
    for (size_t i = 0; i < height.size(); i++)
    {
        /* code */
        while(!st.empty() && height[st.top()] < height[i])
        {
            int curr = st.top();
            st.pop();
            if(st.empty()) break;

            int left = st.top();
            int right = static_cast<int>(i);

            int h = min(height[left], height[right]) - height[curr];

            int w = right - left - 1;

            max_area += (h * w);

        }
        st.push(i);
    }
    return max_area;
}

string multiply(string num1, string num2)
{
    long long first_num = 0, first_digitals = 1;
    for (int i = (num1.size() - 1); i >= 0; i--)
    {
        /* code */
        int one_digital = (num1[i] - '0');

        first_num += first_digitals*one_digital;
        first_digitals *= 10;
    }
    
    long long second_num = 0, second_digitals = 1;
    for (int i = (num2.size() - 1); i >= 0; i--)
    {
        /* code */
        int second_digital = (num2[i] - '0');

        second_num += second_digitals*second_digital;
        second_digitals *= 10;
    }

    long long result = first_num * second_num;
    if(result == 0 || result < 0)
        return "0";

    string ret;
    while (result)
    {
        /* code */
        char digitals= result % 10 + '0';
        ret.insert(ret.begin(), digitals);
        result /= 10;
    }
    

    return ret;
}

vector<vector<int>> permute(vector<int> &nums)
{
    vector<vector<int> > result;
    Recursive_permute(result, nums, 0);
    return result;
}

void Recursive_permute(vector<vector<int>> &result, vector<int> &nums, int start)
{
    int nums_len = static_cast<int>(nums.size());
    if(start == nums_len)
    {
        result.push_back(nums);
        return;
    }

    for (int i = start; i < nums_len; i++)
    {
        /* code */
        swap(nums[i], nums[start]);
        Recursive_permute(result, nums, start + 1);
        swap(nums[i], nums[start]);
    }
}

vector<vector<int>> permuteUnique(vector<int> &nums)
{
    sort(nums.begin(), nums.end());
    //////////////////////////////////////////
    vector<bool> record_used(nums.size(), false);
    vector<int> current;
    vector<vector<int> > results;
    recursive_permuteUnique(nums, results, current, record_used);
    return results;
}

void recursive_permuteUnique(const vector<int>& nums, vector<vector<int> >& result, vector<int>& current, vector<bool>& record_used)
{
    int nums_len = static_cast<int>(nums.size());
    int current_len = static_cast<int>(current.size());
    if(current_len == nums_len)
    {
        result.emplace_back(current);
        return;
    }

    for (int i = 0; i < nums_len; i++)
    {
        if(record_used[i] || (i > 0 && nums[i] == nums[i - 1] && !record_used[i - 1])) continue;
        record_used[i] = true;
        current.push_back(nums[i]);
        recursive_permuteUnique(nums, result, current, record_used);
        current.pop_back();
        record_used[i] = false;
    }
}

void rotate(vector<vector<int> > &matrix)
{
    int n = (int)matrix.size();
    //A[j][i] == A[i][j]
    for (size_t i = 0; i < n; i++)
    {
        /* code */
        for (size_t j = i; j < n; j++)
        {
            /* code */
            swap(matrix[i][j], matrix[j][i]);
        }
    }
    //A[j][n - 1 - i] = A[j][i]
    for (size_t i = 0; i < n; i++)
    {
        reverse(matrix[i].begin(), matrix[i].end());
    }
}

vector<vector<string>> groupAnagrams(vector<string> &strs)
{
    vector<vector<string> > ans;
    map< multiset<char>, vector<string> > str_index;

    int size = (int)strs.size();
    for (size_t i = 0; i < size; i++)
    {
        /* code */
        string tmp = strs.at(i);
        multiset<char> characters = multiset<char>(tmp.begin(), tmp.end());
        if(str_index.find(characters) == str_index.end())
        {
            vector<string> element;
            element.push_back(tmp);
            str_index.insert(pair<multiset<char>, vector<string> >(characters, element));
            continue;
        }
        str_index[characters].push_back(tmp);
    }

    for (auto iter = str_index.begin(); iter != str_index.end(); iter++)
    {
        /* code */
        ans.push_back(iter->second);
    }

    return ans;
}

double myPow(double x, int n)
{
    function<double(char, double, double)> digital_cal = [](char method, double a, double b){
        if(method == '*')
            return a*b; 
        if(method == '/')
            return a/b;
        return 1.0;
    };
    long long N = n;
    char method = '*';
    if(N < 0)
    {
        N = -N;
        method = '/';
    }
    double ans = 1.0;
    for (size_t i = 0; i < N; i++)
    {
        /* code */
        ans = digital_cal(method, ans, x);
    }
    return ans;
}

vector<vector<string> > solveNQueens(int n)
{
    vector<vector<string> > ans;
    vector<string> queen(n, string(n, ' '));
    back_queens(ans, queen, 0, n);
    return ans;
}
bool judge_queens(const vector<string>&queens, const int& n, int i, int j)
{
    //cols
    for (int k = 0; k < i; k++)
    {
        if(queens[k][j] == 'Q')
        {
            return false;
        }
    }
    //left top
    for(int row = i, col = j; row >= 0 && col < n; row--, col++)
    {
        if(queens[row][col] == 'Q')
        {
            return false;
        }
    }

    for(int row = i, col = j; row >= 0 && col >= 0; row--, col--)
    {
        if(queens[row][col] == 'Q')
        {
            return false;
        }
    }
    return true;
}
bool back_queens(vector<vector<string> > &queens, vector<string>& queen, int row, int n)
{
    if(row == n)
    {
        queens.push_back(queen);
        return true;
    }
    bool satisfied_queen = false;
    for (int col = 0; col < n; col++)
    {
        if(judge_queens(queen, n, row, col))
        {
            queen[row][col] = 'Q';
            satisfied_queen = back_queens(queens, queen, row + 1, n) || satisfied_queen;
            queen[row][col] = '.';
        }
    }    
    return satisfied_queen;
}

// void clearBuffer(vector<vector<char>> &buffer)
// {
//     for (int y = 0; y < buffer.size(); y++) {
//         for (int x = 0; x < buffer[y].size(); x++) {
//             buffer[x][y] = ' ';
//         }
//     }
// }

// void displayBuffer(vector<vector<char>> &buffer)
// {
//     for (int y = 0; y < buffer.size(); y++) {
//         for (int x = 0; x < buffer[y].size(); x++) {
//             cout << buffer[x][y];
//         }
//         cout << endl;
//     }
// }

// void draw3DHeart(vector<vector<char>> &buffer, double angle)
// {
//     clearBuffer(buffer);
//     int width = (int)buffer[0].size(), height = (int)buffer.size();
//     for (double t = 0; t < 2 * M_PI; t += 0.05) {
//         for (double theta = 0; theta < 2 * M_PI; theta += 0.05) {
//             // 三维方程：x(t, θ) = 16sin³(t), y(t, θ) = 13cos(t) - 5cos(2t) - 2cos(3t) - cos(4t)
//             double x = 16 * pow(sin(t), 3);
//             double y = 13 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(4 * t);
//             double z = cos(theta + angle) * 10;
//             rotatePoint(x, y, angle);
//             int screenX = (int)(width / 2 + x * 1.5); 
//             int screenY = (int)(height / 2 - y);  
//             if (screenX >= 0 && screenX < width && screenY >= 0 && screenY < height) {
//                 buffer[screenX][screenY] = '*';
//             }
//         }
//     }
//     displayBuffer(buffer);
// }

// void rotatePoint(double &x, double &y, double angle)
// {
//     double newX = x * cos(angle) - y * sin(angle);
//     double newY = x * sin(angle) + y * cos(angle);
//     x = newX;
//     y = newY;
// }

vector<int> spiralOrder(vector<vector<int> > &matrix)
{
    vector<int> ans;
    int left = 0, right = (int)matrix[0].size() - 1;
    int top = 0, bottom = (int)matrix.size() - 1;
    while (true)
    {
        //left -> right
        for (size_t i = left; i <= right; i++)
        {
            ans.emplace_back(matrix[top][i]);
        }
        top++;
        //top -> bottom
        for (size_t i = top; i <= bottom; i++)
        {
            ans.emplace_back(matrix[i][right]);
        }
        right--;
        //right -> left
        for (size_t i = right; i >= left; i--)
        {
            ans.emplace_back(matrix[bottom][i]);
        }
        //bottom -> top
        for (size_t i = --bottom; i >= top; i--)
        {
            ans.emplace_back(matrix[i][left]);
        }
        if(left > right) break;
    }
    return ans;
}

void generate_jc2bank(const char *path)
{
    rapidxml::xml_document<> doc;
    rapidxml::xml_node<>* rot = doc.allocate_node(rapidxml::node_pi,doc.allocate_string("xml version='1.0' encoding='utf-8'"));
    doc.append_node(rot);

    int cnt = 1;
    for (int i = 5; i <= 68; i++, cnt++)
    {
        if(cnt == 9) cnt = 1;

        rapidxml::xml_node<>* node = doc.allocate_node(rapidxml::node_element, "JC", nullptr);
        char* cable_id = doc.allocate_string(("J" + to_string(i + 50)).c_str());
        node->append_attribute(doc.allocate_attribute("Name", cable_id));
        char* origName = doc.allocate_string(("JS" + to_string(i)).c_str());
        node->append_attribute(doc.allocate_attribute("OrigName", origName));

        rapidxml::xml_node<>* child_node = doc.allocate_node(rapidxml::node_element, "Bank", nullptr);
        char* bank_id = doc.allocate_string(to_string(110 + cnt).c_str());
        child_node->append_attribute(doc.allocate_attribute("id", bank_id));
        child_node->append_attribute(doc.allocate_attribute("type", "serdes"));

        char* loc_id = doc.allocate_string(("X0Y" + to_string((cnt - 1)*4)).c_str());
        child_node->append_attribute(doc.allocate_attribute("loc", loc_id));

        node->append_node(child_node);

        doc.append_node(node);
    }
    

    ofstream outfile(path, ios::out);
    outfile << doc;
    outfile.close();

    doc.clear();
}

int uniquePaths(int m, int n)
{
    long long route_num = 1;
    for (int i = 1; i <= m - 1; i++)
    {
        route_num *= ((n - 1 + i) / i);
    }
    return route_num;

    // vector<vector<int> > dcp(m, vector<int>(n, 1));
    // for (int i = 1; i < m; i++)
    // {
    //     for (int j = 1; j < n; j++)
    //     {
    //         /* code */
    //         dcp[i][j] = dcp[i - 1][j] + dcp[i][j - 1];
    //     }
    // }
    // return dcp[m - 1][n - 1];
}
int minPathSum(vector<vector<int>> &grid)
{
    if(grid.empty()) return 0;

    int m = grid.size();
    int n = grid[0].size();

    vector<vector<int> > dcp(m, vector<int>(n, 0));
    dcp[0][0] = grid[0][0];
    for (int i = 1; i < m; i++)
    {
        dcp[i][0] = dcp[i - 1][0] + grid[i][0];
    }
    for (int i = 1; i < n; i++)
    {
        dcp[0][i] = dcp[0][i - 1] + grid[0][i];
    }

    int min_route_value = dcp[0][0];
    for (int i = 1; i < m; i++)
    {
        for(int j = 1; j < n; j++)
        {
            dcp[i][j] = min(dcp[i - 1][j] + grid[i][j], dcp[i][j - 1] + grid[i][j]);
        }
    }
    return dcp[m - 1][n - 1];
}

bool isNumber(string s)
{   
    size_t i = 0;
    bool seen_num = false, seen_E = false, seen_dot = false;
    if(s[i] == '+' || s[i] == '-') i++; 

    for(; i < s.size(); i++)
    {
        if(isdigit(s[i]))
            seen_num = true;
        else if(s[i] == '.')
        {
            if(seen_E || seen_dot) return false;
            seen_dot = true;
        }
        else if(s[i] == 'e' || s[i] == 'E')
        {
            if(!seen_num || seen_E) return false;
            seen_E = true;
            seen_num = false;
        }
        else if(s[i] == '+' || s[i] == '-')
        {
            if(s[i - 1] != 'e' && s[i - 1] != 'E') return false;
        }
        else
            return false;
    }
    return seen_num;
}

vector<int> plusOne(vector<int> &digits)
{
    bool enlarge_size = false;
    for (int i = (int)digits.size() - 1; i >= 0; i--)
    {
        if(digits[i] < 9)
        {
            digits[i]++;
            enlarge_size = false;
            break;
        }
        digits[i] = 0;
        enlarge_size = true;
    }
    if(enlarge_size)
    {
        digits.emplace(digits.begin(), 1);
    }
    return digits;
}

vector<vector<int>> subsets(vector<int> &nums)
{
    vector<vector<int> > ans;
    int nums_size = (int)nums.size();
    int combin_cnt = (1 << nums_size);
    for (int i = 0; i < combin_cnt; i++)
    {
        vector<int> result;
        for (int j = 0; j < nums_size; j++)
        {
            if(i & (1 << j))
            {
                result.push_back(nums[j]);
            }    
        }
        ans.push_back(result);
    }
    // vector<int> result;
    // vector<vector<int> > ans;
    // back_subsets(nums, ans, result, 0);
    return ans;
}

void back_subsets(const vector<int> &nums, vector<vector<int> > &ans, vector<int> &result, int index)
{
    if((int)result.size() <= (int)nums.size())
    {
        ans.push_back(result);
    }
    for (int i=index; i<(int)nums.size(); i++)
    {
        result.push_back(nums[i]);
        back_subsets(nums, ans, result, i+1);
        result.pop_back();
    }
}

ListNode *deleteDuplicates(ListNode *head)
{
    ListNode* pre_node = new ListNode(0);
    pre_node->next = head;

    ListNode* pre_ptr = pre_node;
    ListNode* ptr = head;
    while (ptr)
    {
        bool is_duplicate = false;
        while (ptr->next && ptr->val == ptr->next->val)
        {
            ptr = ptr->next;
            is_duplicate = true;
        }
        if(is_duplicate)
        {
            pre_ptr->next = ptr->next;
            // pre_node = ptr->next;
        }
        else
        {
            pre_ptr = ptr;
        }
        ptr = ptr->next;
    }
    return pre_node->next;
}

int largestRectangleArea(vector<int> &heights)
{
    heights.emplace_back(0);
    
    int max_area = 0;
    stack<int> idx_st;
    for (int i = 0; i < (int)heights.size(); i++)
    {
        while (!idx_st.empty() && heights[i]<heights[idx_st.top()])
        {
            int height = heights[idx_st.top()];

            idx_st.pop();
            int width = (idx_st.empty()? i:i - idx_st.top() - 1);

            max_area = max(max_area, width*height);
        }
        idx_st.push(i);
    }
    return max_area;
}

int maximalRectangle(vector<vector<char>> &matrix)
{
    int row = (int)matrix.size();
    int col = (int)matrix[0].size();

    int max_area = 0;
    vector<int> heights(col, 0);
    for (int i = 0; i < row; i++)
    {
        for (int j=0; j<(int)matrix[i].size(); j++)
        {
            heights[j] = (matrix[i][j] == '1'? heights[j] + 1 : 0);
        }
        max_area = max(max_area, largestRectangleArea(heights));
    }
    return max_area;
}

bool isScramble(string s1, string s2)
{
    bool flag = isScramble(s2, s1, 0);
    return flag;
}

bool isScramble(const string &s2, string &matched, int index)
{
    if(s2 == matched)
    {
        return true;
    }

    for (int i=index; i<(int)matched.size(); i++)
    {
        swap(matched[i], matched[index]);
        if(isScramble(s2, matched, i + 1)) return true;
        swap(matched[i], matched[index]);
    }
    return false;
}

vector<int> grayCode(int n)
{
    int cnt = (1<<n);
    // vector<int> ans = {0};
    // for (int i = 1; i < cnt; i++)
    // {
    //     if(i%2)
    //         ans.push_back(ans[i-1]^1);
    //     else
    //         ans.push_back(ans[i-1]^((ans[i-1]&(-ans[i-1]))<<1));
    // }
    // return ans;

    vector<int> ans;
    for (int i = 0; i < cnt; i++)
    {
        ans.push_back(i^(i>>1));
    }
    return ans;
}

vector<vector<int>> subsetsWithDup(vector<int>& nums)
{
    sort(nums.begin(), nums.end());
    vector<int> ret;
    vector<vector<int> > ans;
    back_subsetsWithDup(ans, ret, nums, 0);
    // int cnt = (1<<(int)nums.size());
    // for (int i = 0; i < cnt; i++)
    // {
    //     vector<int> ret;
    //     for (int j=0; j<(int)nums.size(); j++)
    //     {       
    //         if((i^(i>>1)) ^ j)
    //         {
    //             ret.push_back(nums[j]);
    //         }
    //     }
    //     ans.push_back(ret);
    // }
    return ans;
}

void back_subsetsWithDup(vector<vector<int>> &ans, vector<int> &ret, const vector<int> &nums, int index)
{
    // if(find(ans.begin(), ans.end(), ret) == ans.end())
    ans.push_back(ret);

    for (int i=index; i<(int)nums.size(); i++)
    {
        if(i>index && nums[i]==nums[i-1]) continue;
        ret.push_back(nums[i]);
        back_subsetsWithDup(ans, ret, nums, i+1);
        ret.pop_back();
    }
}

ListNode* reverseBetween(ListNode* head, int left, int right)
{
    ListNode* dummy = new ListNode(0, head);
    ListNode* prev = dummy;

    for (int i = 1; i < left; ++i) {
        prev = prev->next;
    }

    ListNode* start = prev->next;
    ListNode* then = start->next;

    for (int i = 0; i < right - left; ++i) {
        start->next = then->next;
        then->next = prev->next;
        prev->next = then;
        then = start->next;
    }

    return dummy->next;
}

void selection_sort(vector<int>& nums)
{
    for (int i=0; i<(int)nums.size(); i++)
    {
        int min_index = i;
        for (int j=i+1; j<(int)nums.size(); j++)
        {
            min_index = nums[j] > nums[min_index]? min_index:j;
        }
        swap(nums[i], nums[min_index]);
    }
}

void bubble_sort(vector<int> &nums)
{
    for (int i=0; i<(int)nums.size() - 1; i++)
    {
        for (int j=0; j<(int)nums.size() - i - 1; j++)
        {
            if(nums[j] > nums[j+1])
            {
                swap(nums[j+1], nums[j]);
            }
        }
    }
}

void read_probe_list(const char *path)
{
    fstream file(path, ios::in);
    if(!file.is_open())
    {
        cout << "file cannot open" << "\n";
        return;
    }
    string line;
    vector<ProbeSignal> probe_vec;
    while (getline(file, line))
    {
        if(line=="") continue;
        vector<string> nets;
        istringstream ss(line);
        
        string net;
        while (ss >> net)
        {
            nets.push_back(net);
        }
        if(nets.size() < 2) continue;
        deal_with_net(nets, probe_vec);
    }
    auto ret = calculate_net_cnt(probe_vec);
    cout << "The calculated quantity of probe signals: " << ret << "\n";
}
void deal_with_net(const vector<string>& nets, vector<ProbeSignal>& probe_vec)
{
    if(nets[0] != "NET") return;
    string signal = nets[1];

    string lsb = "0";
    string rsb = "0";
    string Nm_singal;

    string charactor = "";
    for (int i = 0; i < (int)signal.size(); i++)
    {
        char C = signal[i];
        if(C != '[' && C != ']' && C != ':')
        {
            charactor.push_back(C);
            continue;
        }
        if(C == '[')
            Nm_singal = charactor;
        else if(C == ':')
            rsb = charactor;
        else if(C == ']')
            lsb = charactor; 

        charactor.clear();   
    }
    bool is_bus = true;
    if(!charactor.empty()) 
    {
        Nm_singal = charactor;
        is_bus = false;
    }
    ProbeSignal probe_signal = ProbeSignal(Nm_singal, stoi(lsb), stoi(rsb), is_bus);
    probe_vec.emplace_back(probe_signal);
}
int calculate_net_cnt(const vector<ProbeSignal> &probe_vec)
{
    cout << "all probe signal: " << endl;
    int cnt = 0;
    for (int i=0; i<(int)probe_vec.size(); i++)
    {
        if(probe_vec[i].is_bus)
        {
            int left  = probe_vec[i].lsb;
            int right = probe_vec[i].rsb;
            if(left > right)
            {
                left  = probe_vec[i].rsb;
                right = probe_vec[i].lsb; 
            }
            for (int j=left; j<=right; j++)
            {
                cout << "\t" <<probe_vec[i].signal << "[" << j << "]" << "\n";
                cnt++;
            }
        }
        else
        {
            cout << "\t" << probe_vec[i].signal << "\n";
            cnt++;
        }
    }
    return cnt;
}
int bit_counts(int N)
{
    int calculate_cnt = 0;
    while (N)
    {
        int only_one = N&(-N);
        N ^= only_one;   
        calculate_cnt++;
    }
    return calculate_cnt;
}
void process_sort(vector<int> &nums, int left, int right)
{
    if(left >= right) return;

    int mid = left + ((right - left) >> 1);
    process_sort(nums, left, mid);
    process_sort(nums, mid + 1, right);
    merge(nums, left, mid, right);
}
void mergeSort(vector<int> &nums)
{
    // int length = (int)nums.size() - 1;
    // process_sort(nums, 0, length);
    int max_size = 1;
    int length = (int)nums.size();
    while (max_size < length)
    {        
        int left = 0;
        while(left < length)
        {
            int mid = left + max_size - 1;
            if(mid>=length)
                break;

            int right = min(mid + max_size, length - 1);

            merge(nums, left, mid, right);
            left += (max_size << 1);
        }
        max_size <<= 1;
    }
}
void merge(vector<int> &nums, int left, int mid, int right)
{
    int i = left;
    int j = mid + 1;
    int idx = 0;
    vector<int> helps(right - left + 1);
    while (i<=mid && j<=right)
    {
        if(nums[i] > nums[j])
            helps[idx++] = nums[j++];
        else
            helps[idx++] = nums[i++];
    }
    while (i<=mid)
    {
        helps[idx++] = nums[i++];
    }
    
    while (j<=right)
    {
        helps[idx++] = nums[j++];
    }
    
    for (int k = 0; k < (int)helps.size(); k++)
    {
        nums[k+left] = helps[k];   
    }
}
bool HeapUseArray::push(int val)
{
    if(size >= limit)
    {
        cout << "Heap is full, and size is "<< size << "\n";
        return false;
    }
    arr[size] = val;

    int index = size;
    int parent = (index - 1) >> 1;
    while (index>0 && arr[index]>arr[parent])
    {
        swap(arr[index], arr[parent]);
        index = parent;
        parent = (index - 1) >> 1;
    }

    size++;
    return true;
}
int HeapUseArray::pop()
{
    if(size == 0)
    {
        cout << "Heap is empty!" << "\n";
        return INT32_MIN;
    }
    int val = arr[0];
    swap(arr[0], arr[--size]);

    int index = 0;
    int left = (index<<1) + 1;
    while (left<size)
    {
        int large = ((left+1<size) && arr[left+1]>arr[left])? left+1:left;
        large = arr[large] > arr[index]? large:index;
        if(large == index)
            break;

        swap(arr[index], arr[large]);
        index = large;
        left = (index<<1) + 1;
    }
    return val;
}
int HeapUseArray::length()
{
    return size;
}
bool HeapUseArray::is_empty()
{
    return (size == 0);
}

void heapSort(vector<int> &arr)
{
    int size = (int)arr.size();
    //generate max heap
    for (int i = size - 1; i >= 0; i--)
    {
        heap_insert(arr, i);
    }
    //update max heap
    for (int i = (size - 1); i >= 0; i--)
    {
        swap(arr[0], arr[i]);
        heap_update(arr, i, 0);
    }
}
void heap_insert(vector<int> &arr, int index) //down to up
{
    int parent = (index - 1) >> 1;
    while (index>0 && arr[index]>arr[parent])
    {
        swap(arr[index], arr[parent]);
        index = parent;
        parent = (index - 1) >> 1;
    }
}
void heap_update(vector<int> &arr, int n, int index) //up to down
{
    int left = (index << 1) + 1;
    while (left < n)
    {
        int large = (left + 1) < n && arr[left+1]>arr[left]? left+1:left;
        large = arr[large] > arr[index]? large:index;
        if(large == index)
            break;
        
        swap(arr[large], arr[index]);
        
        index = large;
        left = (index << 1) + 1;
    }
}
PrefixTree::~PrefixTree()
{
    free_node(root);
}
void PrefixTree::insert(const string &str)
{
    if(str.empty()) return;
    this->root->pass++;

    Node* root_ptr = this->root;
    for (int i = 0; i < (int)str.size(); i++)
    {
        char index = str.at(i);
        if(root_ptr->next[index] == nullptr)
        {
            root_ptr->next[index] = new Node();
        }

        root_ptr = root_ptr->next[index];
        root_ptr->pass++;
    }
    root_ptr->end++;
}
int PrefixTree::search(const string &str)
{
    if(str.empty()) return 0;
    
    Node* root_ptr = root;
    for (int i = 0; i < (int)str.size(); i++)
    {
        char index = str.at(i);
        if(root_ptr->next[index] == nullptr)
        {
            return 0;
        }
        root_ptr = root_ptr->next[index];
    }
    return root_ptr->end;
}
int PrefixTree::prefix_match(const string &str)
{
    if(str.empty()) return 0;
    
    Node* root_ptr = root;
    for (int i = 0; i < (int)str.size(); i++)
    {
        char index = str.at(i);
        if(root_ptr->next[index] == nullptr)
        {
            return 0;
        }
        root_ptr = root_ptr->next[index];
    }
    return root_ptr->pass;
}
void PrefixTree::delete_element(const string &str)
{
    if(this->search(str))
    {
        this->root->pass--;

        Node* root_ptr = this->root;
        for (int i = 0; i < (int)str.size(); i++)
        {
            char index = str.at(i);
            if(--root_ptr->next[index]->pass == 0)
            {
                Node* del_ptr = root_ptr->next[index];
                root_ptr->next[index] = nullptr;
                root_ptr->next.erase(index);

                free_node(del_ptr);
                delete del_ptr;
                return;
            }
            root_ptr = root_ptr->next[index];
        }
        root_ptr->end--;
    }
}
void PrefixTree::free_node(Node *del_node)
{
    if(del_node == nullptr) return;
    
    for (auto iter = del_node->next.begin(); iter != del_node->next.end(); iter++)
    {
        if(iter->second != nullptr)
        {
            free_node(iter->second);
            iter->second = nullptr;
        }
    }
    delete del_node;
}

bool palindrome(ListNode *head)
{
    ListNode* ptr = head;

    stack<int> st;
    while (ptr)
    {
        st.push(ptr->val);
        ptr = ptr->next;
    }
    
    ptr = head;
    while (!st.empty())
    {
        int val = st.top();
        st.pop();

        if(ptr->val != val)
            return false;

        ptr = ptr->next;
    }
    return true;
}

bool palindrome_original_list(ListNode *head)
{
    if(head == nullptr || head->next == nullptr)
        return true;

    ListNode* mid_ptr = mid_upmid_list(head);
    
    //reserve
    ListNode* start_ptr = mid_ptr;
    ListNode* then_ptr = start_ptr->next;

    mid_ptr->next = nullptr;
    while (then_ptr)
    {
        ListNode* next_ptr = then_ptr->next;
        then_ptr->next = start_ptr;

        start_ptr = then_ptr;
        then_ptr = next_ptr;
    }

    //compare
    ListNode* left  = head;
    ListNode* right = start_ptr;
    while (left && right)
    {
        if(left->val != right->val)
            return false;

        left = left->next;
        right = right->next;   
    }
    //revert
    then_ptr = start_ptr->next;
    start_ptr->next = nullptr;
    while (then_ptr)
    {
        ListNode* next_ptr = then_ptr->next;

        then_ptr->next = start_ptr;
        start_ptr = then_ptr;
        then_ptr = next_ptr;
    }

    return true;
}

ListNode *mid_upmid_list(ListNode *head)
{
    if(head->next->next==nullptr)
        return head;

    ListNode* slow_pointer = head->next;
    ListNode* fast_pointer = head->next->next;

    while (fast_pointer->next && fast_pointer->next->next)
    {
        slow_pointer = slow_pointer->next;
        fast_pointer = fast_pointer->next->next;
    }
    return slow_pointer;
}

ListNode *copy_ListNode(ListNode *head)
{
    if(!head) return nullptr;
    map<ListNode*, ListNode*> list_to_list;
    
    ListNode* ptr = head;
    while (ptr)
    {
        ListNode* node = new ListNode(ptr->val);
        list_to_list.insert(make_pair(ptr, node));
        ptr = ptr->next;
    }

    ptr = head;
    while (ptr)
    {
        list_to_list[ptr]->next = ptr->next? list_to_list[ptr->next]:nullptr;
        list_to_list[ptr]->rand = ptr->rand? list_to_list[ptr->rand]:nullptr;
        ptr = ptr->next;
    }
    return list_to_list[head];
}

ListNode *copy_list_unmap(ListNode *head)
{
    ListNode* pointer = head;
    while (pointer)
    {
        ListNode* node = new ListNode(pointer->val);
        ListNode* cur = pointer;
        pointer = pointer->next;
        
        cur->next = node;
        node->next = pointer;
    }

    pointer = head;
    while (pointer)
    {
        ListNode* copied = pointer->next;
        copied->rand = pointer->rand? pointer->rand->next:nullptr;

        pointer = pointer->next->next;   
    }
    //split
    pointer = head;
    ListNode* copied_list = pointer->next;
    while (pointer)
    {
        ListNode* copied = pointer->next;
        pointer->next = copied->next;
        
        pointer = pointer->next;

        if(!pointer) break;
        copied->next = pointer->next;
    }
    return copied_list;
}

ListNode *check_loop_list(ListNode *head)
{
    if(head == nullptr || head->next == nullptr || head->next->next == nullptr)
        return nullptr;
    
    ListNode* slow_ptr = head->next;
    ListNode* fast_ptr = head->next->next;
    while (fast_ptr->next && fast_ptr->next->next)
    {
        slow_ptr = slow_ptr->next;
        fast_ptr = fast_ptr->next->next;
    }

    fast_ptr = head;
    while (fast_ptr != slow_ptr)
    {
        if(slow_ptr == nullptr)
            break;
        fast_ptr = fast_ptr->next;
        slow_ptr = slow_ptr->next;   
    }
    return slow_ptr;
}

ListNode *double_intersection_list(ListNode *head1, ListNode *head2)
{
    ListNode* loop_1 = check_loop_list(head1);
    ListNode* loop_2 = check_loop_list(head2);

    if(loop_1 == nullptr && loop_2 == nullptr)
    {
        return check_intersection_list(head1, head2);
    }

    if(loop_1 != nullptr && loop_2 != nullptr)
    {
        return double_loop_list(head1, loop_1, head2, loop_2);
    }

    return nullptr;
}

ListNode *check_intersection_list(ListNode *head1, ListNode *head2)
{
    if(head1 == nullptr || head2 == nullptr) return nullptr;

    int list_length = 0;
    ListNode* head1_ptr = head1;
    while (head1_ptr)
    {
        list_length++;    
        head1_ptr = head1_ptr->next;
    }
    
    ListNode* head2_ptr = head2;
    while (head2_ptr)
    {
        list_length--;
        head2_ptr = head2_ptr->next;
    }
    
    ListNode* long_head_curr = (list_length > 0? head1 : head2);
    ListNode* short_head_curr = (long_head_curr == head1? head2: head1);

    int cnt = abs(list_length);

    while (cnt--) long_head_curr = long_head_curr->next;   

    while (long_head_curr != short_head_curr)
    {
        if(long_head_curr == nullptr || short_head_curr == nullptr)
            break;

        long_head_curr = long_head_curr->next;
        short_head_curr = short_head_curr->next;
    }
    return long_head_curr;
}

ListNode *double_loop_list(ListNode *head1, ListNode *loop1, ListNode *head2, ListNode *loop2)
{
    if(loop1 == loop2)
    {
        if(head1 == nullptr || head2 == nullptr) return nullptr;

        int list_length = 0;
        ListNode* head1_ptr = head1;
        while (head1_ptr != loop1)
        {
            list_length++;    
            head1_ptr = head1_ptr->next;
        }
        
        ListNode* head2_ptr = head2;
        while (head2_ptr != loop2)
        {
            list_length--;
            head2_ptr = head2_ptr->next;
        }
        
        ListNode* long_head_curr = (list_length > 0? head1 : head2);
        ListNode* short_head_curr = (long_head_curr == head1? head2: head1);

        int cnt = abs(list_length);

        while (cnt--) long_head_curr = long_head_curr->next;   

        while (long_head_curr != short_head_curr)
        {
            if(long_head_curr == loop1 || short_head_curr == loop1)
                break;

            long_head_curr = long_head_curr->next;
            short_head_curr = short_head_curr->next;
        }
        return long_head_curr;
    }
    else
    {
        ListNode* ptr = loop1;
        while (ptr)
        {
            if(ptr == loop2) return loop1;
            if(ptr->next == loop1) break;

            ptr = ptr->next;
        }
    }
    return nullptr;
}

TreeNode *create_tree_node(const vector<int>& m_list, int null_val)
{
    // if(index >= (int)m_list.size() || m_list[index] == -1)
    //     return nullptr;

    // TreeNode* head = new TreeNode(m_list[index]);

    // head->left = create_tree_node(m_list, (index<<1) + 1);
    // head->right = create_tree_node(m_list, (index<<1) + 2);
    // return head;

    if(m_list.empty()) return nullptr;
    TreeNode* root = new TreeNode(m_list[0]);

    queue<TreeNode*> nodes;
    nodes.push(root);

    int index = 1;
    while (!nodes.empty())
    {
        auto top = nodes.front();
        nodes.pop();

        if(index < (int)m_list.size() )
        {
            if(m_list[index] != null_val)
            {
                top->left = new TreeNode(m_list[index]);
                nodes.push(top->left);
            }
            else
                top->left = nullptr;
        }
        index++;
        if(index < (int)m_list.size())
        {
            if( m_list[index] != null_val)
            {
                top->right = new TreeNode(m_list[index]);
                nodes.push(top->right);
            }
            else
                top->right = nullptr;
        }
        index++;
    }
    return root;
}

void preorder_traversal(TreeNode *head)
{
    if(head == nullptr)
        return;
    
    cout << head->val << "\n";
    preorder_traversal(head->left);
    preorder_traversal(head->right);
}

void midorder_traversal(TreeNode *head)
{
    if(head == nullptr)
        return;
    
    midorder_traversal(head->left);
    cout << head->val << "\n";
    midorder_traversal(head->right);
}

void backorder_traversal(TreeNode *head)
{
    if(head == nullptr)
        return;
    
    backorder_traversal(head->left);
    backorder_traversal(head->right);
    cout << head->val << "\n";
}

void preorder_traversal_use_stack(TreeNode *head)
{
    stack<TreeNode*> st;
    st.push(head);
    
    while (!st.empty())
    {
        TreeNode* node = st.top();
        st.pop();
        cout << node->val << "\n";

        if(node->left != nullptr)
        {
            st.push(node->left);
        }
        if(node->right != nullptr)
        {
            st.push(node->right);
        }
    }
}

void midorder_traversal_use_stack(TreeNode *head)
{
    if(head == nullptr) return;
    stack<TreeNode*> node_stack;

    TreeNode* current = head;
    while (!node_stack.empty() || current != nullptr)
    {
        while (current != nullptr)
        {
            node_stack.push(current);
            current = current->left;
        }
        
        current = node_stack.top();
        node_stack.pop();
        cout << current->val << "\n";

        current = current->right;
    }
}

void backorder_traversal_use_stack(TreeNode *head)
{
    if(head == nullptr) return;
    
    stack<TreeNode*> node_stack;
    TreeNode* current = head;
    node_stack.push(current);

    while (!node_stack.empty())
    {
        TreeNode* node = node_stack.top();
        if(node->left!=nullptr && current!=node->left && current!=node->right)
        {
            node_stack.push(node->left);
        }
        else if(node->right!=nullptr && current!=node->right)
        {
            node_stack.push(node->right);
        }
        else
        {
            cout << node->val << "\n";
            node_stack.pop();

            current = node;
        }
    }
}

int max_breadth_tree(TreeNode *head)
{
    if(head == nullptr) return 0;

    TreeNode* current = head;
    TreeNode* end_node = nullptr;

    QueueUseListMultiType<TreeNode*> tree_nodes;
    tree_nodes.add(head);

    int max_nodes = 0, cnt_nodes_nums = 0;
    while (!tree_nodes.is_empty())
    {
        TreeNode* top = tree_nodes.front_val();
        if(top->left != nullptr)
        {
            tree_nodes.add(top->left);
            end_node = top->left;
        }
        if(top->right !=nullptr)
        {
            tree_nodes.add(top->right);
            end_node = top->right;
        }

        cnt_nodes_nums++;
        if(current == top)
        {
            max_nodes = max(max_nodes, cnt_nodes_nums);
            cnt_nodes_nums = 0;
            current = end_node;
        }
    }
    return max_nodes;
}

int max_breadth_tree_use_map(TreeNode *head)
{
    if(head==nullptr) return 0;
    
    QueueUseListMultiType<TreeNode*> tree_nodes;
    tree_nodes.add(head);
    
    int cnt_nodes_num = 0, node_level_num = 1, max_cnt_nodes = INT32_MIN;
    map<TreeNode*, int> nodes_level;
    nodes_level.insert(make_pair(head, 1));
    while (!tree_nodes.is_empty())
    {
        TreeNode* top = tree_nodes.front_val();
        int current_level = nodes_level[top];
        if(top->left != nullptr)
        {
            tree_nodes.add(top->left);
            nodes_level.insert(make_pair(top->left, current_level + 1));
        }
        if(top->right != nullptr)
        {
            tree_nodes.add(top->right);
            nodes_level.insert(make_pair(top->right, current_level + 1));
        }
        if(current_level == node_level_num)
        {
            cnt_nodes_num++;
        }
        else
        {
            max_cnt_nodes = max(cnt_nodes_num, max_cnt_nodes);
            cnt_nodes_num = 1;
            node_level_num++;
        }
    }
    max_cnt_nodes = max(cnt_nodes_num, max_cnt_nodes);
    return max_cnt_nodes;
}

void serialize_tree_node(TreeNode *head, QueueUseListMultiType<int> &serialized_data)
{
    if(head == nullptr)
    {
        serialized_data.add(INT32_MIN);
        return;
    }
    serialized_data.add(head->val);
    serialize_tree_node(head->left, serialized_data);
    serialize_tree_node(head->right, serialized_data);
}

TreeNode *deserialize_tree_node(QueueUseListMultiType<int> &serialized_data)
{
    if(serialized_data.is_empty()) return nullptr;
    int val = serialized_data.front_val();
    if(val == INT32_MIN) return nullptr;

    TreeNode* node = new TreeNode(val);
    node->left = deserialize_tree_node(serialized_data);
    node->right = deserialize_tree_node(serialized_data);

    return node;
}

bool consecutive_natural_numbers(int N)
{
    for (int i = 1; i <= N; i++)
    {
        int sum = i;
        for(int j = i + 1; j <= N; j++)
        {
            sum += j;
            if(N == sum)
                return true;
        }
    }
    return false;
}

bool consecutive_natural_numbers_meter(int N)
{
    if(N < 3) return false;
    
    return (N & (N - 1));
}

void matrix_traversal(const vector<vector<int> >& matrixs)
{
    int x_row = 0, x_col = 0;
    int y_row = 0, y_col = 0;
    int rows = matrixs.size() - 1;
    int cols = matrixs[0].size() - 1;

    bool read_mode = true;
    while (x_row <= rows && x_col <= cols)
    {
        printf_matrix(matrixs, x_row, x_col, y_row, y_col, read_mode);
        x_row = (x_col == cols? x_row+1:x_row);
        x_col = (x_col == cols? x_col:x_col+1);

        y_col = (y_row == rows? y_col+1:y_col);
        y_row = (y_row == rows? y_row:y_row+1);

        read_mode = !read_mode;
    }
}

void printf_matrix(const vector<vector<int>> &matrixs, int x_r, int x_c, int y_r, int y_c, bool read_mode)
{
    if(read_mode)
    {
        while (y_r>=x_r && y_c<=x_c)
        {
            cout << matrixs[y_r][y_c] << " ";  
            y_r--;
            y_c++;
        }
    }
    else
    {
        while (x_r<=y_r && x_c>=y_c)
        {
            cout << matrixs[x_r][x_c] << " ";  
            x_r++;
            x_c--;
        }
    }
    cout.flush();
}

void matrix_rotate_traversal(const vector<vector<int>> &matrixs)
{
    int top = 0, bottom = matrixs.size() - 1;
    int left = 0, right = matrixs[0].size() - 1;
    while (top<=bottom && left<=right)
    {
        if(top<=bottom)
        {
            for (int i=left; i<=right; i++)
            {
                cout << matrixs[top][i] << " ";  
                cout.flush(); 
            }
            top++;
        }
        if(left<=right)
        {
            for (int i=top; i<=bottom; i++)
            {
                cout << matrixs[i][right] << " ";
                cout.flush();
            }
            right--;
        }
        if(top<=bottom)
        {
            for (int i=right; i>=left; i--)
            {
                cout << matrixs[bottom][i] << " ";
                cout.flush();
            }
            bottom--;
        }
        if(left<=right)
        {
            for (int i=bottom; i>=top; i--)
            {
                cout << matrixs[i][left] << " ";
                cout.flush();
            }
            left++;
        }
    }

}

void attain_meet_times(vector<Meet> &meets)
{
    int times = process_meet_times(meets, 0, 0);
    cout << times << "\n";
}

int process_meet_times(const vector<Meet> &meets, int index, int end_time)
{
    if(index >= (int)meets.size())
        return 0;

    int skip = process_meet_times(meets, index+1, end_time);
    int select = 0;
    if(meets[index].start >= end_time)
    {
        select = 1 + process_meet_times(meets, index+1, meets[index].end);
    }
    return max(skip, select);
}

int attain_meet_times_use_greedy(vector<Meet> &meets)
{
    auto compare_meets = [=](const Meet& m1, const Meet& m2) -> bool {
        return m1.end < m2.end;
    };
    sort(meets.begin(), meets.end(), compare_meets);

    int times = 0;
    int end_time = 0;
    for (int i=0; i<(int)meets.size(); i++)
    {
        if(meets[i].start >= end_time)
        {
            times++;
            end_time = meets[i].end;
        }
    }
    return times;
}

void light_place(const string &lights)
{
    set<int> indexs;
    int light_num = process_light_place(lights, 0, indexs);
    cout << light_num << "\n";
}

int process_light_place(const string &lights, int idx, set<int>& indexs)
{
    // if(index >= (int)lights.size())
    // {
    //     return 0;
    // }
    // if(lights[index] == 'X')
    //     return process_light_place(lights, index+1);
    
    // cout << index << "\n";
    // return (1 + process_light_place(lights, index+2));
    if(idx == (int)lights.size())
    {
        for (int i = 0; i<(int)lights.size(); i++)
        {
            if(lights[i] == '.')
            {
                if(!indexs.count(i)&&!indexs.count(i-1)&&!indexs.count(i+1))
                    return INT_MAX;
            }
        }
        return (int)indexs.size();
    }

    int no_dot = process_light_place(lights, idx+1, indexs);
    int has_dot = INT_MAX;
    if(lights[idx] == '.')
    {
        indexs.insert(idx);
        has_dot = process_light_place(lights, idx+1, indexs);
        indexs.erase(idx);
    }
    return min(no_dot, has_dot);
}

int light_place_use_greedy(const string &lights)
{
    int lights_cnt = 0;
    
    int idx = 0;
    while(idx < (int)lights.size())
    {
        char C = lights[idx];
        if(C == 'X')
        {
            idx++;
            continue;
        }
        lights_cnt++;
        if(idx + 1>(int)lights.size()) break;
        if(lights[idx + 1] == 'X')
            idx += 2;
        else
            idx += 3;
        
    }
    return lights_cnt;
}

int attain_max_profit(const vector<Program> &programs, const int& k, int cost)
{
    auto sort_cost = [=](const Program& program1, const Program& program2) -> bool
    {
        return (program1.cost > program2.cost);
    };
    auto sort_profit = [=](const Program& program1, const Program& program2) -> bool
    {
        return (program1.profit<program2.profit);
    };

    priority_queue<Program, vector<Program>, decltype(sort_cost)> min_cost(sort_cost);
    priority_queue<Program, vector<Program>, decltype(sort_profit)> max_profit(sort_profit);
    for (int i=0; i<(int)programs.size(); i++)
    {
        min_cost.push(programs[i]);
    }
    int iteration = k;
    while (iteration--)
    {
        while (!min_cost.empty() && min_cost.top().cost<=cost)
        {
            max_profit.push(min_cost.top());
            min_cost.pop();
        }
        if(max_profit.empty()) break;
        
        cost += max_profit.top().profit;
        max_profit.pop();
    } 
    return cost;
}

template <class T>
UnionSearch<T>::UnionSearch(vector<T> datas)
{
    for (int i=0; i<(int)datas.size(); i++)
    {
        UnionNode<T>* node = new UnionNode<T>(datas[i]);
        nodes.insert(make_pair(datas[i], node));
        set_size.insert(make_pair(node, 1));
        set_parent.insert(make_pair(node, node));
    }
}

template <class T>
UnionNode<T> *UnionSearch<T>::find_father(UnionNode<T> *node)
{
    stack<UnionNode<T>*> stack_nodes;

    UnionNode<T>* current = node;
    while (current!=set_parent[current])
    {
        stack_nodes.push(current);
        current = set_parent[current];
    }
    while (!stack_nodes.empty())
    {
        UnionNode<T>* top = stack_nodes.top();
        stack_nodes.pop();

        set_parent[top] = current;
    }
    return current;
}

template <class T>
bool UnionSearch<T>::is_same_union(T A, T B)
{
    if(!nodes.count(A) || !nodes.count(B))
        return false;

    UnionNode<T>* A_node = nodes[A];
    UnionNode<T>* B_node = nodes[B]; 
    return find_father(A_node) == find_father(B_node);
}

template <class T>
void UnionSearch<T>::set_same_union(T A, T B)
{
    if(!nodes.count(A) || !nodes.count(B) || is_same_union(A, B))
        return;

    UnionNode<T>* A_node = nodes[A];
    UnionNode<T>* B_node = nodes[B]; 

    int A_size = set_size[A_node];
    int B_size = set_size[B_node];

    if(A_size <= B_size)
    {
        UnionNode<T>* B_parent = find_father(B_node);
        UnionNode<T>* A_parent = find_father(A_node);
        set_parent[A_node] = B_parent;

        set_size[B_parent]++;
        set_size.erase(A_parent);
    }
    else
    {
        UnionNode<T>* B_parent = find_father(B_node);
        UnionNode<T>* A_parent = find_father(A_node);
        set_parent[B_node] = A_parent;

        set_size[A_parent]++;
        set_size.erase(B_parent);
    }
}

template <class T>
int UnionSearch<T>::size()
{
    return (int)set_size.size();
}

template <class T>
inline Edge<T>::Edge(shared_ptr<GrapNode<T> > from_point, shared_ptr<GrapNode<T> > to_point, double weight)
{
    this->from_point = from_point;
    this->to_point   = to_point;
    this->weight     = weight;
}

template <class T>
double Edge<T>::get_weight()
{
    return this->weight;
}

template <class T>
shared_ptr<GrapNode<T>> Edge<T>::get_from_point()
{
    return from_point.lock();
}

template <class T>
shared_ptr<GrapNode<T>> Edge<T>::get_to_point()
{
    return to_point.lock();
}

template <class T>
bool Edge<T>::operator<(const Edge &other) const
{
    return weight<other.weight;
}

template <class T>
GrapNode<T>::GrapNode(T value)
{
    this->value = value;
    out_degree = 0;
    in_degree  = 0;
}

template <class T>
inline T GrapNode<T>::get_value() const
{
    return this->value;
}

template <class T>
bool GrapNode<T>::add_next_node(const shared_ptr<GrapNode<T>> &next)
{
    if(next.get() == nullptr) return false;
    nearby_nodes.push_back(next);
    return true;
}

template <class T>
bool GrapNode<T>::add_edge(const shared_ptr<Edge<T>> &edge)
{
    if(edge.get() == nullptr) return false;
    nearby_edges.push_back(edge);
    return true;
}

template <class T>
bool GrapNode<T>::add_out_degree()
{
    out_degree++;
    return true;
}

template <class T>
bool GrapNode<T>::add_in_degree()
{
    in_degree++;
    return true;
}

template <class T>
vector<shared_ptr<Edge<T> > > &GrapNode<T>::get_nearby_edges()
{
    return nearby_edges;
}

template <class T>
vector<shared_ptr<GrapNode<T>>> &GrapNode<T>::get_nearby_nodes()
{
    return nearby_nodes;
}

template <class T>
inline Graph<T>::Graph(const vector<tuple<T, T, double> > &datas)
{
    for (int i = 0; i < (int)datas.size(); i++)
    {
        T from = get<0>(datas[i]);
        T to = get<1>(datas[i]);
        double weight = get<2>(datas[i]);
        if(!nodes.count(from))
        {
            shared_ptr<GrapNode<T> > from_node(new GrapNode<T>(from));
            nodes.insert(make_pair(from, from_node));
        }
        if(!nodes.count(to))
        {
            shared_ptr<GrapNode<T> > to_node(new GrapNode<T>(to));
            nodes.insert(make_pair(to, to_node));
        }
        
        nodes[from]->add_next_node(nodes[to]);
        nodes[from]->add_out_degree();

        nodes[to]->add_in_degree();

        shared_ptr<Edge<T> > edge(new Edge<T>(nodes[from], nodes[to], weight));
        edges.push_back(edge);

        nodes[from]->add_edge(edge);
    }
}

template <class T>
Graph<T>::~Graph()
{
    edges.clear();
}

template <class T>
void Graph<T>::breadth_first_traversal(T value)
{
    if(!nodes.count(value)) return;
    shared_ptr<GrapNode<T> > node = nodes[value];

    set<shared_ptr<GrapNode<T> > > visited;
    deque<shared_ptr<GrapNode<T> > > deque_nodes;
    deque_nodes.push_back(node);
    visited.insert(node);
    while (!deque_nodes.empty())
    {
        shared_ptr<GrapNode<T> > top = deque_nodes.front();
        deque_nodes.pop_front();

        cout << top->get_value() << "\n";

        vector<shared_ptr<GrapNode<T> > >& child_nodes = top->get_nodes();
        for (int i=0; i<(int)child_nodes.size(); i++)
        {
            shared_ptr<GrapNode<T> > child_node = child_nodes[i];
            if(visited.count(child_node)) continue;
            
            visited.insert(child_node);
            deque_nodes.push_back(child_node);
        }
    }
}

template <class T>
void Graph<T>::depth_first_traversal(T value)
{
    if(!nodes.count(value)) return;
    shared_ptr<GrapNode<T> > node = nodes[value];
    
    cout << value << "\n";
    stack<shared_ptr<GrapNode<T> > > stack_nodes;
    set<shared_ptr<GrapNode<T> > > visited;
    stack_nodes.push(node);
    visited.insert(node);
    while (!stack_nodes.empty())
    {
        shared_ptr<GrapNode<T> > top = stack_nodes.top();
        stack_nodes.pop();

        vector<shared_ptr<GrapNode<T> > >& child_nodes = top->get_nodes();
        for (int i=0; i<(int)child_nodes.size(); i++)
        {
            if(visited.count(child_nodes[i])) continue;
            stack_nodes.push(top);
            stack_nodes.push(child_nodes[i]);
            visited.insert(child_nodes[i]);
            cout << child_nodes[i]->get_value() << "\n";
            break;
        }
    }
}

template <class T>
void Graph<T>::topological_sort(vector<T>& ans)
{
    queue<shared_ptr<GrapNode<T> > > queue_nodes;
    map<shared_ptr<GrapNode<T> >, int> map_degrees;
    for (auto iter=nodes.begin(); iter!=nodes.end(); iter++)
    {
        shared_ptr<GrapNode<T> > node = iter->second;
        if(node->get_in_quantity() == 0)
        {
            queue_nodes.push(node);
        } 
        map_degrees.insert(make_pair(node, node->get_in_quantity()));
    }

    while (!queue_nodes.empty())
    {
        shared_ptr<GrapNode<T> > top = queue_nodes.front();
        queue_nodes.pop();

        ans.push_back(top->get_value());
        // cout << top->get_value() << "\n";
        vector<shared_ptr<GrapNode<T> > >& child_nodes = top->get_nodes();
        for (int i=0; i<(int)child_nodes.size(); i++)
        {
            if(--map_degrees[child_nodes[i]] == 0)
            {
                queue_nodes.push(child_nodes[i]);
            }
        }
    }
    // check whether circuit loop
    // cout << "hello world" << "\n";
}

template <class T>
void Graph<T>::minimum_span_subgraph_kruskal(vector<Edge<T>> &update_edges)
{
    auto ComSort = [](const shared_ptr<Edge<T> >& edge1, const shared_ptr<Edge<T> >& edge2){
        return edge1->get_weight()>edge2->get_weight();
    };

    priority_queue<shared_ptr<Edge<T> >, vector<shared_ptr<Edge<T> > >, decltype(ComSort)> small_poil(ComSort);
    for (int i=0; i<(int)edges.size(); i++)
    {
        small_poil.push(edges[i]);
    }
    
    unique_ptr<UnionSearch<T> > m_union(new UnionSearch<T>(get_values()));
    while (!small_poil.empty())
    {
        shared_ptr<Edge<T> > top = small_poil.top();
        small_poil.pop();

        T from_value = top->get_from_point()->get_value();
        T to_value = top->get_to_point()->get_value();

        if(!m_union->is_same_union(from_value, to_value))
        {
            update_edges.push_back(*top.get());
            m_union->set_same_union(from_value, to_value);
        }
    }
}

template <class T>
void Graph<T>::minimum_span_subgraph_prim(vector<Edge<T>> &update_edges)
{
    auto ComSort = [](const shared_ptr<Edge<T> >& edge1, const shared_ptr<Edge<T> >& edge2){
        return edge1->get_weight()>edge2->get_weight();
    };
    priority_queue<shared_ptr<Edge<T> >, vector<shared_ptr<Edge<T> > >, decltype(ComSort)> small_poil(ComSort);

    set<shared_ptr<GrapNode<T> > > selected_nodes;
    for (auto iter=nodes.begin(); iter!=nodes.end(); iter++)
    {
        shared_ptr<GrapNode<T> > node = iter->second;
        if(!selected_nodes.count(node))
        {
            selected_nodes.insert(node);

            vector<shared_ptr<Edge<T> > >& node_edges = node->get_nearby_edges();
            for (int i = 0; i < (int)node_edges.size(); i++)
            {
                small_poil.push(node_edges[i]);
            }
            while (!small_poil.empty())
            {
                shared_ptr<Edge<T> > top = small_poil.top();
                small_poil.pop();

                shared_ptr<GrapNode<T> > from_node = top->get_from_point();
                shared_ptr<GrapNode<T> > to_node = top->get_to_point();

                if(selected_nodes.count(to_node) && selected_nodes.count(from_node)) continue;

                shared_ptr<GrapNode<T> > candidate;
                if(selected_nodes.count(from_node))
                    candidate = to_node;
                else
                    candidate = from_node;

                selected_nodes.insert(candidate);
                update_edges.push_back(*top.get());

                vector<shared_ptr<Edge<T> > >& top_edges = candidate->get_nearby_edges();
                for (int i=0; i<(int)top_edges.size(); i++)
                {
                    small_poil.push(top_edges[i]);
                }
            }
        }
    }
}

template <class T>
map<shared_ptr<GrapNode<T> >, double> Graph<T>::minimum_generation_path(T value)
{
    shared_ptr<GrapNode<T> > node = nodes[value];

    map<shared_ptr<GrapNode<T> >, double> records;
    records.insert(make_pair(node, 0));

    set<shared_ptr<GrapNode<T> > > selected_nodes;
    shared_ptr<GrapNode<T> > current = find_min_nearbynode(selected_nodes, records);
    while (current.get() != nullptr)
    {
        double distance = records[current];
        vector<shared_ptr<Edge<T> > >& m_edges = current->get_nearby_edges();
        for (int i = 0; i < (int)m_edges.size(); i++)
        {
            shared_ptr<GrapNode<T> > to_node = m_edges[i]->get_to_point();
            if(!records.count(to_node))
            {
                records[to_node] = distance + m_edges[i]->get_weight();
            }
            else
            {
                records[to_node] = min(distance + m_edges[i]->get_weight(), records[to_node]);
            }
        }
        selected_nodes.insert(current);
        current = find_min_nearbynode(selected_nodes, records);
    }
    return records;
}

template <class T>
vector<T> Graph<T>::get_values()
{
    vector<T> values;
    for (auto iter=nodes.begin(); iter!=nodes.end(); iter++)
    {
        values.push_back(iter->first);
    }
    return values;
}

template <class T>
shared_ptr<GrapNode<T>> Graph<T>::find_min_nearbynode(const set<shared_ptr<GrapNode<T> > > &selected_nodes, const map<shared_ptr<GrapNode<T> >, double> &records)
{
    double distance = INT_MAX;
    shared_ptr<GrapNode<T> > current;
    for (auto iter=records.begin(); iter!=records.end(); iter++)
    {
        if(!selected_nodes.count(iter->first)&&iter->second < distance)
        {
            current = iter->first;
            distance = iter->second;
        }
    }
    return current;
}

void hanio(int N, const string &src, const string &dst, const string &aid)
{
    if(N == 1)
    {
        cout << src << " move " << dst << "\n";
        return;
    }
    hanio(N-1, src, aid, dst);
    cout << src << " move " << dst << "\n";
    hanio(N-1, aid, dst, src);
}

void queens(int N)
{
    // auto start = chrono::steady_clock::now();
    // vector<int> matrix(N); 
    // auto count = queens_process(matrix, 0);
    // cout << count << " need time: ";
    // auto end = chrono::steady_clock::now();
    // auto duration_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    // cout << duration_time << "\n";

    auto start = chrono::steady_clock::now();
    vector<bitset<8> > queen(N);
    int limit = (1<<N) - 1;
    auto calculate_total = queens_process_use_bit(limit, 0, 0, 0, 0, queen);
    cout << calculate_total << " need time: ";
    auto end = chrono::steady_clock::now();
    auto duration_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << duration_time << "\n";
}

int queens_process(vector<int> &matrix, int row)
{
    if((int)matrix.size() == row)
    {
        return 1;
    }

    int N = 0;
    for (int col = 0; col < (int)matrix.size(); col++)
    {
        matrix[row] = col;
        if(is_valid_queen(matrix, row, col))
        {
            N += queens_process(matrix, row+1);
        }
    }
    return N;
}

bool is_valid_queen(const vector<int> &matrix, int i, int j)
{
    for (int k = 0; k < i; k++)
    {
        if(matrix[k]==j || abs(k - i)==abs(matrix[k] - j))
            return false;
    }
    return true;
}

int queens_process_use_bit(const int &limit, int col_limit, int left_limit, int right_limit, int row, vector<bitset<8> >& queen_pos)
{
    if(col_limit == limit)
    {
        for (int i = 0; i < (int)queen_pos.size(); i++)
        {
            cout << queen_pos[i].to_string() << "\n";
        }
        cout << "\n\n\n";
        return 1;
    }
    int count = 0;
    int pos = limit & (~(left_limit | right_limit | col_limit));
    while (pos!=0)
    {
        int left_pos = pos & (~pos + 1);
        pos ^= left_pos;

        bitset<8> row_c(left_pos);
        queen_pos[row] = row_c;
        
        count += queens_process_use_bit(limit, col_limit|left_pos, (left_limit|left_pos)<<1, (right_limit|left_pos)>>1, row+1, queen_pos);
    }
    return count;
}

void min_sticker(const vector<string>& stickers, const string& rest)
{
    vector<vector<int> > tran_stickers((int)stickers.size(), vector<int>(26));
    for (int i = 0; i < (int)stickers.size(); i++)
    {
        for (int j = 0; j < (int)stickers[i].size(); j++)
        {
            tran_stickers[i][stickers[i][j] - 'a']++;
        }
    }
    vector<int> tran_rest(26);
    for (int i=0; i<(int)rest.size(); i++)
    {
        tran_rest[rest[i] - 'a']++;
    }
    unordered_map<string, int> dcp;
    auto ret = traverse_min_sticker(tran_stickers, tran_rest, dcp);
    cout << ret << "\n";
}

int traverse_min_sticker(const vector<vector<int> >& tran_stickers, vector<int> tran_rest, unordered_map<string, int>& dcp)
{
    bool has_rest = false;
    for (int i = 0; i < (int)tran_rest.size(); i++)
    {
        if(tran_rest[i]>0)
        {
            has_rest = true;
            break;
        }
    }
    if(!has_rest) return 0;
    
    auto serlize = [=](const vector<int>& m_rest) -> string {
        string result = "";
        for (int i=0; i<(int)m_rest.size(); i++)
        {
            if(m_rest[i] == 0) continue;
            for (int j = 0; j < m_rest[i]; j++)
            {
                result.push_back(i + 'a');   
            }
        }
        return result;
    };
    string serlize_str = serlize(tran_rest);
    if(dcp.count(serlize_str)) return dcp[serlize_str];

    int calculate_total = INT_MAX;
    for (int i=0; i<(int)tran_stickers.size(); i++)
    {
        bool has_comm = false;
        for (int j=0; j<(int)tran_stickers[i].size(); j++)
        {
            if(tran_stickers[i][j]>0&&tran_rest[j]>0)
            {
                has_comm = true;
                break;
            }
        }
        if(!has_comm) continue;

        vector<int> new_rest = tran_rest;
        for (int j = 0; j < tran_rest.size(); j++)
        {
            new_rest[j] = max(new_rest[j] - tran_stickers[i][j], 0);
        }
        
        int count = traverse_min_sticker(tran_stickers, new_rest, dcp);
        if(count != INT_MAX)
        {
            calculate_total = min(calculate_total, count+1);
        }
    }
    
    dcp[serlize_str] = calculate_total;
    return dcp[serlize_str];
}

int numTrees(int n)
{
    if(n <= 1) return 1;
    int total = 0;

    for (int i = 1; i <= n; i++)
    {
        total+= numTrees(i-1)*numTrees(n-i);
    }
    return total;
}

int numTrees_use_dp(int n)
{
    vector<int> dp(n+1);
    dp[0] = 1;

    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= i; j++)
        {
            dp[i] += dp[j - 1]*dp[i - j];
        }
    }
    return dp[n];
}

vector<TreeNode*> generateTrees(int n)
{
    return traverse_tree(1, n);
}

vector<TreeNode *> traverse_tree(int left, int right)
{
    vector<TreeNode*> result;
    if(left>right)
    {
        result.push_back(nullptr);
        return result;
    }
    for (int i = left; i <= right; i++)
    {
        vector<TreeNode*> left_trees = traverse_tree(left, i-1);
        vector<TreeNode*> right_trees = traverse_tree(i+1, right);

        for (auto& left_node:left_trees)
        {
            for(auto& right_node:right_trees)
            {
                TreeNode* node = new TreeNode(i);
                node->left = left_node;
                node->right = right_node;
                result.push_back(node);
            }
        }
    }
    return result;
}

bool isValidBST(TreeNode *root)
{
    // return traverse_validBST(root, LONG_LONG_MIN, LONG_LONG_MAX);
    stack<TreeNode*> nodes;

    long long pre_val = LONG_LONG_MIN;
    TreeNode* current = root;
    while (current!=nullptr || !nodes.empty())
    {
        while (current!=nullptr)
        {
            nodes.push(current);
            current = current->left;
        }
        
        current = nodes.top();
        nodes.pop();

        if(pre_val >= current->val)
            return false;

        pre_val = current->val;

        current = current->right;
    }
    return true;
}

bool traverse_validBST(TreeNode *root, long long min_val, long long max_val)
{
    if(root == nullptr) return true;

    if(root->val <= min_val || root->val >= max_val) return false;

    return traverse_validBST(root->left, min_val, root->val)&&traverse_validBST(root->right, root->val, max_val);
}

void recoverTree(TreeNode *root)
{
    // stack<TreeNode*> nodes;

    // int pre_val = INT_MIN;
    // TreeNode* current = root;
    // TreeNode* pre_node = nullptr;
    // TreeNode* first = nullptr;
    // TreeNode* second = nullptr;
    // while (!nodes.empty() || current!=nullptr)
    // {
    //     while (current!=nullptr)
    //     {
    //         nodes.push(current);
    //         current = current->left;
    //     }
    //     current = nodes.top();
    //     nodes.pop();

    //     if(pre_val >= current->val)
    //     {
    //         if(first == nullptr)
    //         {
    //             first = pre_node;
    //         }
    //         second = current;
    //     }
    //     pre_node = current;
    //     pre_val = current->val;

    //     current = current->right;
    // }
    
    // if(first != nullptr)
    // {
    //     swap(first->val, second->val);
    // }

    TreeNode* pre_node = nullptr;
    TreeNode* first = nullptr;
    TreeNode* second = nullptr;
    traverse_recoverTree(root, first, second, pre_node);
    if(first != nullptr)
    {
        swap(first->val, second->val);
    }
}

void traverse_recoverTree(TreeNode *root, TreeNode *first, TreeNode *second, TreeNode *pre_node)
{
    if(!root) return;

    traverse_recoverTree(root->left, first, second, pre_node);

    if(pre_node!=nullptr && root->val<=pre_node->val)
    {
        if(first == nullptr)
        {
            first = pre_node;
        }
        second = root;
    }
    pre_node = root;
    traverse_recoverTree(root->right, first, second, pre_node);
}

bool isSameTree(TreeNode *p, TreeNode *q)
{
    // if(p == nullptr && q == nullptr) return true;

    // if(p == nullptr || q == nullptr) return false;


    // if(p->val != q->val) return false;

    // return isSameTree(p->left, q->left) & isSameTree(p->right, q->right);

    stack<pair<TreeNode*, TreeNode*> > nodes;
    nodes.push(make_pair(p, q));

    while (nodes.empty())
    {
        auto top = nodes.top();
        nodes.pop();

        if(top.first == nullptr && top.second == nullptr) continue;


        if(top.first ==nullptr || top.second ==nullptr || top.first->val != top.second->val)
        {
            return false;
        }

        nodes.push(make_pair(top.first->right, top.second->right));
        nodes.push(make_pair(top.first->left, top.second->left));
    }
    return true;
}

bool isSymmetric(TreeNode *root)
{
    return traverse_isSymmetric(root->left, root->right);


//     stack<pair<TreeNode*, TreeNode*> > nodes;
//     nodes.push(make_pair(root->left, root->right));

//     while (!nodes.empty())
//     {
//         auto top = nodes.top();
//         nodes.pop();

//         if(top.first == nullptr&&top.second == nullptr) continue;
        
//         if(top.first == nullptr || top.second == nullptr)
//         {
//             return false;
//         }
//         if(top.first->val != top.second->val)
//             return false;

//         nodes.push(make_pair(top.first->left, top.second->right));
//         nodes.push(make_pair(top.first->right, top.second->left));
//     }
    
//     return true;
}

bool traverse_isSymmetric(TreeNode *p, TreeNode *q)
{
    if(p == nullptr && q == nullptr) return true;

    if(p == nullptr || q == nullptr) return false;

    if(p->val != q->val) return false;
    
    return traverse_isSymmetric(p->left, q->right)&&traverse_isSymmetric(p->right, q->left);
}

vector<vector<int>> levelOrder(TreeNode *root)
{
    if(root == nullptr) return vector<vector<int> >();

    queue<TreeNode*> nodes;
    nodes.push(root);

    TreeNode* current = root;
    TreeNode* end_node = nullptr;

    vector<int> row_content;
    vector<vector<int> > result;
    while (!nodes.empty())
    {
        TreeNode* top = nodes.front();
        nodes.pop();

        if(top->left!=nullptr)
        {
            nodes.push(top->left);
            end_node = top->left;
        }
        if(top->right!=nullptr)
        {
            nodes.push(top->right);
            end_node = top->right;
        }

        row_content.push_back(top->val);
        if(top == current)
        {
            result.push_back(row_content);

            row_content.clear();
            current = end_node;
        }
    }
    return result;
    // vector<vector<int>> ans;
    // queue<std::pair<int, TreeNode*>> q;
    // TreeNode* cur;
    // q.push(std::pair(0, root));
    // while(!q.empty())
    // {
    //     cur = q.front().second;
    //     int level = q.front().first;
    //     q.pop();
    //     if(cur == nullptr)
    //         continue;
    //     q.push(std::pair(level + 1, cur->left));
    //     q.push(std::pair(level + 1, cur->right));
    //     while (ans.size() <= level) 
    //         ans.push_back({});

    //     ans[level].push_back(cur->val);
    // }
    // return ans;
}

vector<vector<int>> zigzagLevelOrder(TreeNode *root)
{
    if(root == nullptr) return vector<vector<int> >();

    queue<TreeNode*> nodes;
    nodes.push(root);
    
    TreeNode* current = root;
    TreeNode* end_node = nullptr;

    bool iter_sort = false;
    vector<vector<int>> ans;
    vector<int> row_data;
    while (!nodes.empty())
    {
        auto top = nodes.front();
        nodes.pop();

        if(top->left != nullptr)
        {
            nodes.push(top->left);
            end_node = top->left;
        }
        if(top->right != nullptr)
        {
            nodes.push(top->right);
            end_node = top->right;
        }

        row_data.push_back(top->val);
        if(current == top)
        {
            if(iter_sort)
                reverse(row_data.begin(), row_data.end());

            ans.push_back(row_data);
            iter_sort = !iter_sort;
            row_data.clear();
            current = end_node;
        }
    }
    return ans;
}

int maxDepth(TreeNode *root)
{
    if(root == nullptr) return 0;
    int left_count = maxDepth(root->left) + 1;
    int right_count = maxDepth(root->right) + 1;

    return max(left_count, right_count);
}

TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder)
{
    // if(preorder.empty()) return nullptr;
    // TreeNode* root = new TreeNode(preorder[0]);

    // stack<TreeNode*> nodes;
    // nodes.push(root);
    // int in_index = 0;
    // for (int i = 1; i < preorder.size(); i++)
    // {
    //     auto top = nodes.top();
    //     if(top->val != inorder[in_index])
    //     {
    //         top->left = new TreeNode(preorder[i]);
    //         nodes.push(top->left);
    //     }
    //     else
    //     {
    //         while (!nodes.empty()&&nodes.top()->val == inorder[in_index])
    //         {
    //             top = nodes.top();
    //             nodes.pop();
    //             in_index++;
    //         }
    //         top->right = new TreeNode(preorder[i]);
    //         nodes.push(top->right);
    //     }
    // }
    // return root;
    ///////////////////////////////////////////////////////////////////////////////
    if(preorder.empty()) return nullptr;
    map<int, int> in_map;
    for (int i=0; i<(int)inorder.size(); i++)
    {
        in_map.insert(make_pair(inorder[i], i));
    }
    return traverse_buildTree(preorder, 0, (int)preorder.size()-1, inorder, 0, (int)inorder.size() - 1, in_map);
}

TreeNode *traverse_buildTree(const vector<int> &preorder, int pre_start, int pre_end, const vector<int> &inorder, int in_start, int in_end, const map<int, int> &in_pos)
{
    if(pre_start>pre_end || in_start>in_end) return nullptr;

    TreeNode* root = new TreeNode(preorder[pre_start]);

    int step_size = in_pos.find(preorder[pre_start])->second;

    int left_num = step_size - in_start;

    root->left = traverse_buildTree(preorder, pre_start+1, pre_start+left_num, inorder, in_start, step_size - 1, in_pos);
    root->right = traverse_buildTree(preorder, pre_start+left_num+1, pre_end, inorder, step_size+1, in_end, in_pos);

    return root;
}

TreeNode *buildTree_in_post(vector<int> &inorder, vector<int> &postorder)
{
    if(postorder.empty()) return nullptr;

    int length = (int)postorder.size() - 1;
    TreeNode* root = new TreeNode(postorder[length]);

    stack<TreeNode*> nodes;
    nodes.push(root);

    int in_order_index = (int)inorder.size() - 1;
    for (int i=length - 1; i>=0; i--)
    {
        auto top = nodes.top();
        if(top->val != inorder[in_order_index])
        {
            top->right = new TreeNode(postorder[i]);
            nodes.push(top->right);
        }
        else
        {
            while (!nodes.empty()&&nodes.top()->val==inorder[in_order_index])
            {
                top = nodes.top();
                nodes.pop();
                in_order_index--;
            }
            top->left = new TreeNode(postorder[i]);
            nodes.push(top->left);
        }
    }
    return root;
}

bool isBalanced(TreeNode * root)
{
    if(root == nullptr) return true;
    stack<TreeNode*> nodes;
    map<TreeNode*, int> levels;

    TreeNode* lasted_node = nullptr;
    TreeNode* current = root;
    while (!nodes.empty() || current!=nullptr)
    {
        while (current!=nullptr)
        {
            nodes.push(current);
            current = current->left;
        }
        
        current = nodes.top();
        if(current->right!=nullptr && lasted_node!=current->right)
        {
            current = current->right;
        }
        else
        {
            nodes.pop();

            int left_level = (current->left!=nullptr? levels[current->left]:0);
            int right_level = (current->right!=nullptr? levels[current->right]:0);

            if(abs(left_level-right_level)>1)
                 return false;

            levels[current] = max(left_level, right_level) + 1;

            lasted_node = current;

            current = nullptr;
        }
    }
    return true;
    // return abs(maxDepth(root->left) - maxDepth(root->right))<=1;
}

int minDepth(TreeNode *root)
{
    if(root == nullptr) return 0;

    if(root->left == nullptr) return minDepth(root->right)+1;
    if(root->right == nullptr) return minDepth(root->left)+1;

    return (min(minDepth(root->left), minDepth(root->right)) + 1);
}

vector<vector<int> > pathSum(TreeNode *root, int targetSum)
{
    // vector<vector<int> > ans;
    // vector<int> one;

    // traverse_hasPathSum(root, targetSum, one, ans);
    // return ans;
    vector<vector<int> > ans;
    if(root == nullptr) return ans;

    stack<tuple<TreeNode*, int, vector<int> > > nodes;
    nodes.push(make_tuple(root, root->val, vector<int>{root->val}));
    while (!nodes.empty())
    {
        auto top = nodes.top();
        nodes.pop();
        TreeNode* top_node = get<0>(top);
        int sum            = get<1>(top);

        if(top_node->left==nullptr&&top_node->right==nullptr&&sum==targetSum)
        {
            ans.push_back(get<2>(top));
        }

        if(top_node->right != nullptr)
        {
            vector<int> one    = get<2>(top);
            one.push_back(top_node->right->val);    
            nodes.push(make_tuple(top_node->right, sum+top_node->right->val, one));
        }
        
        if(top_node->left != nullptr)
        {
            vector<int> one    = get<2>(top);
            one.push_back(top_node->left->val);
            nodes.push(make_tuple(top_node->left, sum+top_node->left->val, one));
        }
    }
    return ans;
}

void traverse_hasPathSum(TreeNode *root, int targetSum, vector<int> &one, vector<vector<int>> &ans)
{
    if(root == nullptr) return;

    one.push_back(root->val);
    if(root->left==nullptr && root->right==nullptr && targetSum==root->val)
    {
        ans.push_back(one);
        return;
    }
    traverse_hasPathSum(root->left, targetSum - root->val, one, ans);
    traverse_hasPathSum(root->right, targetSum - root->val, one, ans);
    one.pop_back();
}

void flatten(TreeNode *root)
{
    if(root == nullptr) return;
    TreeNode* update_tree = new TreeNode();

    stack<TreeNode*> nodes;
    nodes.push(root);

    while (!nodes.empty())
    {
        auto top = nodes.top();
        nodes.pop();

        if(top->right!=nullptr)
        {
            nodes.push(top->right);
        }

        if(top->left != nullptr)
        {
            nodes.push(top->left);
        }
        update_tree->right = top;
        update_tree->left = nullptr;
        update_tree = update_tree->right;
    }

    cout << "hello" << "";
}

int numDistinct(string s, string t)
{
    int s_length = (int)s.size();
    int t_length = (int)t.size();
    
    vector<vector<long long> > dcp(s_length+1, vector<long long>(t_length+1, 0));

    for (int i = 0; i < s_length; i++)
    {
        dcp[i][0] = 1;
    }
    
    for (int i = 1; i <= s_length; i++)
    {
        for (int j = 1; j <= t_length; j++)
        {
            dcp[i][j] = dcp[i-1][j];
            if(s[i-1] == t[j-1])
            {
                dcp[i][j] += dcp[i-1][j-1];
            }                
        }
    }

    return dcp[s_length][t_length];
    // string ans;
    // auto calculate_total = traverse_numDistinct(s, t, 0, ans);
    // return calculate_total;
}

int traverse_numDistinct(const string &s, const string &t, int index, string &ans)
{
    if((int)ans.size()>(int)t.size()) return 0;

    if(ans == t) return 1;
    if(index == (int)s.size()) return 0;

    int total = 0;
    for (int i = index; i < (int)s.size(); i++)
    {
        ans.push_back(s[i]);
        total += traverse_numDistinct(s, t, i+1, ans);
        ans.pop_back();
    }
    return total;
}

NODE *create_NODE_tree(const vector<int> &data, const int& null_val)
{
    if((int)data.size() == 0) return nullptr;
    
    NODE* root = new NODE(data[0]);

    queue<NODE*> nodes;
    nodes.push(root);

    int index = 1;
    while (!nodes.empty())
    {
        auto top = nodes.front();
        nodes.pop();

        if(index<(int)data.size())
        {
            if(data[index] != null_val)
            {
                top->left = new NODE(data[index]);
                nodes.push(top->left);
            }
            else
                top->left = nullptr;

            index++;
        }
        if(index>=(int)data.size()) break;

        if(index<(int)data.size())
        {
            if(data[index] != null_val)
            {
                top->right = new NODE(data[index]);
                nodes.push(top->right);
            }
            else
                top->right = nullptr;

            index++;
        }
    }
    return root;
}

NODE *connect(NODE *root)
{
    if(root == nullptr) return nullptr;

    queue<NODE*> nodes;
    nodes.push(root);

    NODE* last_vistied = root;
    while (!nodes.empty())
    {
        int sz = (int)nodes.size();

        last_vistied = nodes.front();
        for (int i = 0; i < sz; i++)
        {
            auto front = nodes.front();
            nodes.pop();
            if(front->left!=nullptr) nodes.push(front->left);
            if(front->right!=nullptr) nodes.push(front->right);

            if(last_vistied!=front)
            {
                last_vistied->next = front;
                last_vistied = front;
            }
        }
        last_vistied->next = nullptr;
    }
    return root;
}

int minimumTotal(vector<vector<int> > &triangle)
{
    vector<int> dcp = triangle.back();
    for (int i=(int)triangle.size() - 2; i>=0; i--)
    {        
        for (int j=0; j<(int)triangle[i].size(); j++)
        {
            dcp[j] = triangle[i][j] + min(dcp[j], dcp[j+1]);
        }
    }
    return dcp[0];
}

int maxProfit_II(vector<int> &prices)
{
    int profit = 0;
    for (int i=1; i<(int)prices.size(); i++)
    {
        if(prices[i]>prices[i-1])
        {
            profit += prices[i]-prices[i-1];
        }
    }
    return profit;
}

int maxProfit_III(vector<int> &prices) //////////////leetcode 123
{
    int f_sell_p = 0, f_buy_c = INT_MIN;
    int s_sell_p = 0, s_buy_c = INT_MIN;
    
    for (int i=0; i<(int)prices.size(); i++)
    {
        f_buy_c = max(f_buy_c, -prices[i]);
        f_sell_p = max(f_sell_p, f_buy_c+prices[i]);

        s_buy_c = max(s_buy_c, f_sell_p-prices[i]);
        s_sell_p = max(s_sell_p, s_buy_c+prices[i]);
    }
    return s_sell_p;
}

int maxPathSum(TreeNode *root)
{
    int sum = INT_MIN;
    traverse_maxPathSum(root, sum);
    return sum;


    // int sum = 0;

    // stack<TreeNode*> nodes;
    // TreeNode* current = root;
    // while (!nodes.empty() || current!= nullptr)
    // {
    //     while (current!=nullptr)
    //     {
    //         nodes.push(current);
    //         current = current->left;
    //     }
    //     current = nodes.top();
    //     nodes.pop();

    //     int val = current->val;
    //     sum = max(val + sum, val);
    //     current = current->right;
    // }
    // return sum;
}

int traverse_maxPathSum(TreeNode *root, int &sum)
{
    if(root == nullptr) return 0;

    int left_val = max(traverse_maxPathSum(root->left, sum), 0);
    int right_val = max(traverse_maxPathSum(root->right, sum), 0);


    sum = max(sum, root->val+left_val+right_val);

    return root->val + max(left_val, right_val);
}

vector<vector<string> > findLadders(string beginWord, string endWord, vector<string> &wordList)
{
    if(beginWord == endWord) return vector<vector<string> >();
    unordered_map<string, vector<string> > word_pattern;
    auto attain_regex = [&](const string &str) -> void
    {
        for (int i = 0; i < (int)str.size(); i++)
        {
            string tmp = str;
            tmp[i] = '*';
            word_pattern[tmp].push_back(str);
        }
    };

    for (int i = 0; i < (int)wordList.size(); i++)
    {
        attain_regex(wordList[i]);
    }
    attain_regex(beginWord);

    unordered_map<string, int> distances;
    unordered_map<string, vector<string>> process_pattern;
    queue<string> q;
    q.push(beginWord);
    distances[beginWord] = 0;

    bool find_flag = false;
    while (!q.empty() && !find_flag)
    {
        int level = static_cast<int>(q.size());
        for (int i = 0; i < level; i++)
        {
            string top = q.front();
            q.pop();

            for (int j = 0; j < (int)top.size(); j++)
            {
                string pattern = top;
                pattern[j] = '*';
                if (!word_pattern.count(pattern))
                    continue;

                for (int k = 0; k < (int)word_pattern[pattern].size(); k++)
                {
                    string matched = word_pattern[pattern][k];

                    if (!distances.count(matched))
                    {
                        distances[matched] = distances[top] + 1;
                        process_pattern[matched].push_back(top);
                        q.push(matched);
                    }

                    else if (distances[matched] == distances[top] + 1)
                    {
                        process_pattern[matched].push_back(top);
                    }

                    if(matched == endWord) 
                        find_flag = true;
                }
            }
        }
    }
    vector<string> rets;
    vector<vector<string> > ans;
    if(!distances.count(endWord)) return ans;
    function<void(const string&)> back_track = [&](const string& curr) {
        rets.push_back(curr);
        if(curr == beginWord)
        {
            reverse(rets.begin(), rets.end());
            ans.push_back(rets);
            reverse(rets.begin(), rets.end());
        }
        else
        {
            for (int i = 0; i < (int)process_pattern[curr].size(); i++)
            {
                back_track(process_pattern[curr][i]);
            }
        }
        rets.pop_back();
    };
    if(distances.count(endWord))
    {
        back_track(endWord);
    }
    return ans;
}

int longestConsecutive(vector<int> &nums)
{
    if(nums.empty()) return 0;
    sort(nums.begin(), nums.end());

    int counts = 0, longest_num = 0;
    for(int i = 1; i < (int)nums.size(); i++)
    {
        if(nums[i] - nums[i-1] == 0) continue;

        if(nums[i] - nums[i-1] == 1)
        {
            counts++;
        }
        else
        {
            counts = 0;
        }
        longest_num = max(longest_num, counts);
    }
    return longest_num + 1;
}

int sumNumbers(TreeNode * root)
{
    stack<pair<TreeNode*, string> > nodes;

    int sum = 0;
    nodes.push(make_pair(root, to_string(root->val)));
    while (!nodes.empty())
    {
        auto top = nodes.top();
        nodes.pop();

        if(top.first->left == nullptr && top.first->right == nullptr)
        {
            sum += stoi(top.second);
        }

        if(top.first->left != nullptr)
        {
            string next_level = top.second + to_string(top.first->left->val);
            nodes.push(make_pair(top.first->left, next_level));
        }
        if(top.first->right != nullptr)
        {
            string next_level = top.second + to_string(top.first->right->val);
            nodes.push(make_pair(top.first->right, next_level));
        }
    }
    return sum;
}

void solve(vector<vector<char> > &board)
{   
    if(board.size() == 0) return;

    int row = static_cast<int>(board.size());
    int col = static_cast<int>(board[0].size());
    queue<pair<int, int> > coordinate;
    auto mark = [&](int i, int j) -> void
    {
        if(i<0 || i>=row || j<0 || j>=col || board[i][j]!='O')
            return;
        board[i][j] = '#';
        coordinate.push({i,j});
    };
    // auto mark
    for (int i=0; i<row; i++)
    {
        mark(i, 0);
        mark(i, col - 1);
    }
    for (int i=0; i<col; i++)
    {
        mark(0, i);
        mark(row - 1, i);
    }

    vector<pair<int, int> > round_all = {{-1,0},{0,1},{0,-1},{1,0}};
    while (!coordinate.empty())
    {
        auto top = coordinate.front();
        coordinate.pop();
        for (int i=0; i<(int)round_all.size(); i++)
        {
            mark(top.first+round_all[i].first, top.second+round_all[i].second);
        }
    }
    
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            if(board[i][j] == 'O') board[i][j] = 'X';
            if(board[i][j] == '#') board[i][j] = 'O';
        }
    }
}

vector<vector<string> > partition(string s)
{
    auto isPalindrome = [&](const string& str, int left, int right) -> bool
    {
        while (left<right)
        {
            if(str.at(left++) != str.at(right--))
                return false;
        }
        return true;
    };
    vector<vector<string> > ans;
    long long cnt = 1<<(static_cast<int>(s.size()) - 1);
    for (int i = 0; i < cnt; i++)
    {
        int start = 0;
        vector<string> rets;
        for (int j = 0; j < (int)s.size() - 1; j++)
        {
            if(i & (1<<j))
            {
                rets.push_back(s.substr(0, i-start+1));
                start = i + 1;
            }
        }
        rets.push_back(s.substr(start));

        ans.push_back(rets);
    }
    return ans;
}

int minCut(string s)
{
    // auto isPalindrome = [&](const string& str, int left, int right) -> bool
    // {
    //     while (left<right)
    //     {
    //         if(str.at(left++) != str.at(right--))
    //             return false;
    //     }
    //     return true;
    // };

    // int start = 0, cut_count = INT_MAX;
    // unsigned long long count_loop = (1 << (static_cast<int>(s.size()) - 1));
    // for (int i = 0; i < count_loop; i++)
    // {
    //     int start = 0;
    //     vector<string> rets;
    //     for (int j = 0; j < (int)s.size() - 1; j++)
    //     {
    //         if(i & (1<<j))
    //         {
    //             rets.emplace_back(s.substr(start, j - start + 1));
    //             start = j + 1;
    //         }    
    //     }
    //     rets.push_back(s.substr(start));

    //     bool is_correct = true;
    //     for (int j = 0; j < (int)rets.size(); j++)
    //     {
    //         if(!isPalindrome(rets[j], 0, (int)rets[j].size() - 1))
    //         {
    //             is_correct = false;
    //             break;
    //         }
    //     }
    //     if(is_correct) cut_count = min(cut_count, (int)rets.size());
    // }
    // return cut_count - 1;
    int s_len = static_cast<int>(s.size());
    vector<vector<bool> > isPad(s_len, vector<bool>(s_len, false));
    for (int i = s_len - 1; i >= 0; i--)
    {
        for (int j = i; j < s_len; j++)
        {
            if(s[i] == s[j] && ( j - i < 2 || isPad[i + 1 ][j - 1]))
                isPad[i][j] = true;
        } 
    }
    
    vector<int> dcp(s_len, INT_MAX);
    for (int i = 0; i < s_len; i++)
    {
        if(isPad[0][i])
            dcp[i] = 0;
        else
        {
            for (int j = 1; j <= i; j++)
            {
                if(isPad[j][i])
                    dcp[i] = min(dcp[i], dcp[j - 1] + 1);
            }
        }
    }
    return dcp[s_len - 1];
}

GraphNode* buildGraph(const vector<vector<int> >& adj) {
    if (adj.empty()) return nullptr;
    int n = (int)adj.size();
    vector<GraphNode*> nodes(n + 1, nullptr); // nodes[1..n]
    for (int i = 1; i <= n; ++i) nodes[i] = new GraphNode(i);
    for (int i = 1; i <= n; ++i) {
        for (int v : adj[i-1]) {
            nodes[i]->neighbors.push_back(nodes[v]);
        }
    }
    return nodes[1]; 
}

GraphNode *cloneGraph(GraphNode *node)
{
    if(node == nullptr) return nullptr;

    unordered_map<GraphNode*, GraphNode*> visited;

    GraphNode* cloneNode = new GraphNode(node->val);
    visited[node] = cloneNode;

    queue<GraphNode*> q;
    q.push(node);
    
    while (!q.empty()) {
        GraphNode* cur = q.front();
        q.pop();

        for (GraphNode* neighbor : cur->neighbors) {
            if (visited.find(neighbor) == visited.end()) {
                visited[neighbor] = new GraphNode(neighbor->val);
                q.push(neighbor);
            }

            visited[cur]->neighbors.push_back(visited[neighbor]);
        }
    }
    return cloneNode;
}

int canCompleteCircuit(vector<int> &gas, vector<int> &cost)
{
    int gas_count = static_cast<int>(gas.size());
    for (int i = 0; i < gas_count; i++)
    {
        bool is_satisfaied = true;
        int rest_gas = 0;
        for (int j = 0; j < gas_count; j++)
        {
            int k = (i + j) % gas_count;

            rest_gas += (gas[k] - cost[k]);

            if(rest_gas < 0)
            {
                is_satisfaied = false;
                break;
            } 
        }
        if(is_satisfaied)
        {
            return i;
        }
    }
    return -1;
}

bool checkPowersOfThree(int n)
{
    int div_n = n;
    while (div_n)
    {
        int remainder = div_n % 3;
        if(remainder == 2)
            return false;

        div_n /= 3;
    }
    return true;
}

int candy(vector<int> &ratings)
{
    // int rate_len = static_cast<int>(ratings.size());
    // vector<int> values(rate_len, 1);
    // for (int i = 1; i < rate_len; i++)
    // {
    //     if(ratings[i] > ratings[i - 1])
    //     {
    //         values[i] = values[i - 1] + 1;
    //     }
    // }

    // for (int i = rate_len - 2; i >= 0; i--)
    // {
    //     if(ratings[i] > ratings[i+1])
    //     {
    //         values[i] = max(values[i + 1] + 1, values[i]);
    //     }
    // }
    
    // int total = 0;
    // for (int i = 0; i < (int)values.size(); i++)
    // {
    //     total += values[i];
    // }
    // return total;

    int rate_len = static_cast<int>(ratings.size());
    int peak = 0, down = 0, up = 0, total = 0;
    for (int i = 1; i < rate_len; i++)
    {
        if(ratings[i] > ratings[i-1])
        {
            up++;
            down = 0;
            peak = up;
            total += up + 1;
        }
        else if(ratings[i] == ratings[i-1])
        {
            up = down = peak = 0;
            total += 1;
        }
        else
        {
            down++;
            up = 0;
            total += 1 + down;

            if(down>peak)
            {
                total += 1;
            }
        }
    }
    return total;
}

bool isPowerOfFour(int n)
{
    return n > 0 && (n & (n - 1)) == 0 && (n & 0x55555555) != 0;
}

int maximum69Number(int num)
{
    string str = to_string(num);
    for (int i = 0; i < (int)str.size(); i++)
    {
        if(str[i] == '6')
        {
            str[i]='9';
            break;
        }
    }
    return stoi(str);
}

int singleNumber(vector<int> &nums)
{
    int exclude_or = 0;
    for (int i = 0; i < (int)nums.size(); i++)
    {
        exclude_or ^= nums[i];
    }
    return exclude_or;
}

int singleNumber2(vector<int> &nums)
{
    int one = 0, two = 0;
    for (int i=0; i<(int)nums.size(); i++)
    {
        one = (nums[i] ^ one) & ~two;
        two = (nums[i] ^ two) & ~one;
    }
    return one;
}

double new21Game(int n, int k, int maxPts)
{
    

    return 0.0;
}

long long zeroFilledSubarray(vector<int> &nums)
{
    int nums_len = static_cast<int>(nums.size());
    int cnt = 0, res = 0;
    for (int i = 1; i < nums_len; i++)
    {
        if(nums[i] == 0)
        {
            cnt++;
            res += cnt;
        }
        else
            cnt = 0;
    }
    return res;
}

int countSquares(vector<vector<int> > &matrix)
{
    int rows = (int)matrix.size();
    if(rows == 0) return 0;
    int cols = (int)matrix[0].size();
    vector<vector<int> > dcps(rows, vector<int>(cols));
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if(matrix[i][j] == 1 && i>0&&j>0)
            {
                dcps[i][j] = 1 + min({dcps[i-1][j-1], dcps[i-1][j], dcps[i][j-1]});
            }
            else
                dcps[i][j] = matrix[i][j];
        }
    }

    int total = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            total += dcps[i][j];
        }
    }
    return total;
}

SpecialNode *copyRandomList(SpecialNode *head)
{
    unordered_map<SpecialNode*, SpecialNode*> node_map;
    SpecialNode* copied_node = new SpecialNode(head->val);
    SpecialNode* ptr = head->next;
    SpecialNode* copied_ptr = copied_node;
    
    node_map.insert(make_pair(head, copied_node));
    while (ptr)
    {
        if(!node_map.count(ptr))
        {
            SpecialNode* node = new SpecialNode(ptr->val);
            node_map[ptr] = node;
        }

        node_map[copied_ptr]->next = node_map[ptr];

        ptr = ptr->next;
    }

    return copied_ptr;
}

double maxAverageRatio(vector<vector<int> > &classes, int extraStudents)
{
    struct ComAverage
    {
        bool operator()(const AverageNode& a1, const AverageNode& a2)
        {
            return a1.gain_ < a2.gain_;
        }
    };

    priority_queue<AverageNode, vector<AverageNode>, ComAverage> m_priorty;
    for (int i = 0; i < (int)classes.size(); i++)
    {
        m_priorty.push({classes[i][0], classes[i][1]});
    }
    
    while (extraStudents--)
    {
        auto top = m_priorty.top();
        m_priorty.pop();

        top.p_ += 1;
        top.t_ += 1;
        top.gain_ = AverageNode::computeGain(top.p_, top.t_);
        
        m_priorty.push(top);
    }

    double total = 0.0;
    while (!m_priorty.empty())
    {
        auto top = m_priorty.top();
        m_priorty.pop();

        total += ((double)top.p_ / (double)top.t_);
    }
    return total / static_cast<double>(classes.size());
}

bool wordBreak(string s, vector<string> &wordDict)
{
    unordered_set<string> check_word(wordDict.begin(), wordDict.end());

    int max_len = 0, s_len = static_cast<int>(s.size());
    for (int i = 0; i < (int)wordDict.size(); i++)
    {
        max_len = max(static_cast<int>(wordDict[i].size()), max_len);
    }
    
    vector<bool> dcp(s_len + 1, false);
    dcp[0] = true;
    for (int i = 1; i <= s_len; i++)
    {
        for (int j = i - 1; j >= max(0, i - max_len); j--)
        {
            if(!dcp[j]) continue;
            
            string split_str = s.substr(j, i - j);
            if(check_word.count(split_str))
            {
                dcp[i] = true;
                break;
            }
        }
    }
    return dcp[s_len];
}

constexpr float exp(float x, int n)
{
    return n == 0 ? 1 :
        n % 2 == 0 ? exp(x * x, n / 2) :
        exp(x * x, (n - 1) / 2) * x;
}

void InsertSort::sort(vector<int> &nums)
{
    std::sort(nums.begin(), nums.end());
}

void QuickSort::sort(vector<int> &nums)
{
    std::sort(nums.begin(), nums.end());
}

void StrategyWay::set_execute_method(unique_ptr<Interface> method)
{
    method_ = std::move(method);
}

void StrategyWay::execute_method(vector<int> &nums)
{
    if(method_)
        method_->sort(nums);
}

const int THREAD_SIZE = 10;
queue<int> works;
condition_variable condi_var;
mutex mtx;

void produce()
{
    for (size_t i = 0; i < 20; i++)    
    {
        unique_lock<mutex> lock(mtx);

        condi_var.wait(lock, []{ return works.size() < THREAD_SIZE; });
        works.push(i);
        cout << "run idx: " << i  << "\n";
        lock.unlock();
        condi_var.notify_one();
    }
}

void comsumer()
{
    while (true)
    {
        unique_lock<mutex> lock(mtx);
        condi_var.wait(lock, []{ return !works.empty(); });
        int data = works.front();
        works.pop();
        cout << "data: " << data << "\n";

        lock.unlock();
        condi_var.notify_one();

        if(data == 19) break;
    }
}

MeomoryPool* MemoryAllocateWay::init_pool(uint32_t block_size, uint32_t pool_size)
{
    MeomoryPool* pool = (MeomoryPool*)malloc(sizeof(MeomoryPool));
    if(pool == NULL)
    {
        perror("malloc:");
        return NULL;
    }

    pool->block_chunk_per = pool_size;
    pool->block_size = (block_size < sizeof(MemoryNode*)?sizeof(MemoryNode*):block_size);
    pool->used = 0;
    pool->chunk = NULL;
    pool->ptr = NULL;
    return pool;
}

bool MemoryAllocateWay::allocate_pool(MeomoryPool *pool)
{
    void* ptr = malloc(pool->block_size*pool->block_chunk_per);
    if(ptr == NULL)
    {
        perror("allocate_pool:");
        return false;
    }
    pool->ptr = (void**)realloc(pool->ptr, sizeof(void*) * (pool->used + 1));

    pool->ptr[pool->used++] = ptr;

    uint8_t* cur = (uint8_t*)ptr;
    for (int i = 0; i < (int)pool->block_chunk_per; i++)
    {
        MemoryNode* memory_node = (MemoryNode*)cur;
        memory_node->next = pool->chunk;
        pool->chunk = memory_node;
        
        cur += pool->block_size;
    }
    return true;
}

void MemoryAllocateWay::free_pool(MeomoryPool *pool)
{
    if(pool == NULL) return;
    for (int i = 0; i < pool->used; i++)
    {
        free(pool->ptr[i]);
    }
    free(pool->ptr);
    free(pool);
}

void* MemoryAllocateWay::single_alloc(MeomoryPool *pool)
{
    if(pool == NULL)
    {
        if(!MemoryAllocateWay::allocate_pool(pool))
        {
            return NULL;
        }
    }

    MemoryNode* space = pool->chunk;
    pool->chunk = space->next;
    return (void*)space;
}

bool MemoryAllocateWay::single_free(MeomoryPool *pool, void *space)
{
    MemoryNode* node = (MemoryNode*)space;
    node->next = pool->chunk;
    pool->chunk = node;
    return true;
}

uint32_t MemoryAllocateWay::value_two_pow(uint32_t value)
{
    --value;
    value |= (value >> 1);
    value |= (value >> 2);
    value |= (value >> 4);
    value |= (value >> 8);
    value |= (value >> 16);
    value |= (value >> 32);
    ++value;
    return value;
}

NonFixedMemoryPool *MemoryAllocateWay::init_nonfix_pool(uint32_t min_size, uint32_t max_size, uint32_t capatity)
{
    MemoryAllocateWay::NonFixedMemoryPool* pool = new MemoryAllocateWay::NonFixedMemoryPool();
    uint32_t m = value_two_pow(min_size);
    uint32_t M = value_two_pow(max_size);
    pool->max_size = M;
    pool->min_size = m;

    uint32_t block_num = 0;
    for (size_t i = m; i <= M; i<<=1) block_num++;
    pool->unfix_blocks = block_num;

    pool->unfix_array = (uint32_t*)malloc(block_num*sizeof(uint32_t));
    pool->fixed_pools = (MemoryAllocateWay::MeomoryPool**)malloc(block_num*sizeof(MemoryAllocateWay::MeomoryPool*));
    if(!pool->unfix_array || !pool->fixed_pools) return nullptr;

    block_num = 0;
    for (size_t i = m; i <= M; i<<=1, block_num++)
    {
        pool->unfix_array[block_num] = i;

        if(MemoryAllocateWay::MeomoryPool* fixed_pool = init_pool(i, capatity))
        {
            allocate_pool(fixed_pool);
            pool->fixed_pools[block_num] = fixed_pool;
            // free(fixed_pool);
        }
        else
        {
            for (size_t i = 0; i <= block_num; i++)
                free(&pool->unfix_array[i]);
            
            for (size_t i = 0; i <= block_num; i++)
                free_pool(pool->fixed_pools[i]);
            
            free(pool->fixed_pools);
            free(pool->unfix_array);
            delete pool;
            pool = nullptr;
            break;
        }
    }
    return pool;
}

void *MemoryAllocateWay::allocate_nonfixed_pool(MemoryAllocateWay::NonFixedMemoryPool *pool, uint32_t size)
{
    if(pool == nullptr) return nullptr;
    if(size>pool->max_size) return nullptr;

    uint32_t real_size = value_two_pow(size);
    if(real_size < pool->min_size) 
        real_size = pool->min_size;

    uint32_t idx = 0;
    for(uint32_t i=pool->min_size; i<=real_size;i<<=1, idx++);

    return single_alloc(pool->fixed_pools[idx]);
}

void MemoryAllocateWay::deallocate_nonfixed_pool(MemoryAllocateWay::NonFixedMemoryPool *pool, void *space, uint32_t size)
{
    if(pool == nullptr || space == nullptr) return;
    if(size > pool->max_size)
    {
        cout << "please input correct size!!!" << "\n";
        return;
    }

    uint32_t real_size = MemoryAllocateWay::value_two_pow(size);
    uint32_t idx = 0;
    for(size_t i=pool->min_size; i<=real_size; i<<=1, idx++);
    single_free(pool->fixed_pools[idx], space);
}

void MemoryAllocateWay::free_nonfixed_pool(MemoryAllocateWay::NonFixedMemoryPool *pool)
{
    uint32_t idx = 0;
    for(size_t i=pool->min_size; i<=pool->max_size; i<<=1, idx++)
    {
        free_pool(pool->fixed_pools[idx]);
        // free(&(pool->fixed_pools[idx]));
    }  
    free(pool->fixed_pools);
    free(pool->unfix_array);
    delete pool;
    pool = nullptr;
}

uint32_t ObjectPool::round_up_to_align(uint32_t value, uint32_t align_size)
{
    if(align_size == 0 || align_size & (align_size - 1) != 0)
        return value;

    return (value + align_size - 1) & ~(align_size - 1);
}

ObjectPool::ObjectAllocate *ObjectPool::pool_init(uint32_t object_size, uint32_t capatity)
{
    if(object_size == 0 || capatity == 0)
        return nullptr;
    if(object_size < sizeof(ObjectPool::ObjectMemory))
    {
        object_size = static_cast<uint32_t>(sizeof(ObjectPool::ObjectMemory));
    }
    uint32_t ailgn = alignof(ObjectPool::ObjectMemory);
    uint32_t stride = ObjectPool::round_up_to_align(object_size, ailgn); 

    ObjectPool::ObjectAllocate* obejct_pool = new ObjectPool::ObjectAllocate();
    obejct_pool->object_size = object_size;
    obejct_pool->capatity    = capatity;
    obejct_pool->buff = nullptr;
    obejct_pool->memory = nullptr;

    void* buffer = malloc(stride*capatity);
    if(buffer == nullptr)
    {
        delete obejct_pool;
        return nullptr;
    }
    obejct_pool->buff = buffer;

    uint8_t* ptr = (uint8_t*)buffer;
    // ObjectPool::ObjectMemory* obj_mem = (ObjectPool::ObjectMemory*)obejct_pool->buff;
    for (uint32_t i = 0; i < capatity; i++)
    {
        ObjectPool::ObjectMemory* obj_mem = reinterpret_cast<ObjectPool::ObjectMemory*>(ptr) ;
        obj_mem->next = obejct_pool->memory;
        obejct_pool->memory = obj_mem;

        ptr += object_size;
    }
    return obejct_pool;
}

void *ObjectPool::pool_allocate(ObjectAllocate *obj_pool)
{
    if(!obj_pool || !obj_pool->memory)
        return nullptr;

    ObjectPool::ObjectMemory* ptr = obj_pool->memory;
    obj_pool->memory = ptr->next;
    return (void*)ptr;
}

void ObjectPool::pool_deallocate(ObjectAllocate *obj_pool, void *obj)
{
    if(!obj_pool || !obj)
        return;

    ObjectPool::ObjectMemory* ptr = (ObjectPool::ObjectMemory*)obj;
    ptr->next = obj_pool->memory;
    obj_pool->memory = ptr;
}

void ObjectPool::destory_pool(ObjectAllocate *obj_pool)
{
    if(!obj_pool) return;

    std::free(obj_pool->buff);
    obj_pool->buff = nullptr;

    delete obj_pool;
    obj_pool = nullptr;
}

typename STL::CustomQueue *STL::init_queue(uint32_t size)
{
    if(size == 0) return nullptr;
    CustomQueue* m_queue = (STL::CustomQueue*)malloc(sizeof(STL::CustomQueue));
    m_queue->head = 0;
    m_queue->tail = 0;
    m_queue->capatity = size;
    m_queue->data = (void**)malloc(sizeof(void*)*size);
    if(m_queue->data == nullptr)
    {
        free(m_queue);
        return nullptr;
    }
    return m_queue;
}

typename STL::CustomQueue *STL::expand_queue(STL::CustomQueue * m_queue)
{
    STL::CustomQueue* expanded_queue = (STL::CustomQueue*)malloc(sizeof(STL::CustomQueue));
    expanded_queue->capatity = m_queue->capatity << 1;
    expanded_queue->head = 0;

    expanded_queue->data = (void**)malloc(sizeof(void*)*expanded_queue->capatity);
    if(expanded_queue->data == nullptr) return nullptr;

    for (uint32_t i = 0; i < queue_size(m_queue); i++)
    {
        expanded_queue->data[i] = m_queue->data[(m_queue->head + i)%m_queue->capatity];
    }
    free(m_queue->data);
    m_queue->capatity = 0;
    m_queue->head = 0;
    m_queue->tail = 0;
    free(m_queue);
    return expanded_queue;
}

void STL::push_front_val(CustomQueue *m_queue, void *value)
{
    if((m_queue->head - 1 + m_queue->capatity) % m_queue->capatity == m_queue->tail)
        m_queue = expand_queue(m_queue);

    m_queue->data[(m_queue->head - 1 + m_queue->capatity) % m_queue->capatity] = value;
    m_queue->head = (m_queue->head - 1 + m_queue->capatity) % m_queue->capatity;
}

void *STL::pop_front_val(CustomQueue *m_queue)
{
    if(m_queue == nullptr)
        return nullptr;

    void* ptr = m_queue->data[m_queue->head];
    m_queue->head = ((m_queue->head + 1) % m_queue->capatity);
    return ptr;
}

void STL::push_back_val(CustomQueue *m_queue, void *value)
{
    if(((m_queue->tail + 1 + m_queue->capatity) % m_queue->capatity) == m_queue->head)
        m_queue = expand_queue(m_queue);

    m_queue->data[m_queue->tail] = value;
    m_queue->tail = (m_queue->tail + 1) % m_queue->capatity;
}

void *STL::pop_back_val(CustomQueue *m_queue)
{
    if(m_queue == nullptr)
        return nullptr;

    void* ptr = m_queue->data[(m_queue->tail - 1) % m_queue->capatity];
    m_queue->tail = (m_queue->tail - 1) % m_queue->capatity;
    return ptr;
}

void STL::destory_queue(CustomQueue *m_queue)
{
    if(m_queue == nullptr) return;
    m_queue->head = 0;
    m_queue->tail = 0;
    m_queue->capatity = 0;
    free(m_queue->data);
    free(m_queue);
}

uint32_t STL::queue_size(CustomQueue *m_queue)
{
    if(m_queue == nullptr) return 0;
    return (m_queue->tail - m_queue->head + m_queue->capatity) % m_queue->capatity;
}

Screen::StoreNode* Screen::freeStore = nullptr;
const int Screen::maxStore = 24;

void *Screen::operator new(size_t m_size)
{
    if(m_size <= sizeof(Screen::StoreNode))
        m_size = sizeof(Screen::StoreNode);

    if(freeStore == nullptr)
    {
        uint8_t* ptr = static_cast<uint8_t*>(::operator new(m_size*maxStore));
        freeStore = reinterpret_cast<Screen::StoreNode*>(ptr);

        Screen::StoreNode* curr = freeStore;
        for (int i = 1; i < maxStore; i++)
        {
            auto next = reinterpret_cast<Screen::StoreNode*>(ptr + i*m_size);
            curr->next = next;
            curr = next;
        }
        curr->next = nullptr;
    }
    Screen::StoreNode* firstStore = freeStore;
    freeStore = freeStore->next;
    return firstStore;
}

void Screen::operator delete(void *ptr) noexcept
{
    Screen::StoreNode* curr = reinterpret_cast<Screen::StoreNode*>(ptr);
    curr->next = freeStore;
    freeStore = curr;
}

AirPlane* AirPlane::freeStore = nullptr;
const int AirPlane::maxCount = 24;

void *AirPlane::operator new(size_t m_size)
{
    if(freeStore == nullptr)
    {
        uint8_t* ptr = static_cast<uint8_t*>(::operator new(m_size*maxCount));
        freeStore = reinterpret_cast<AirPlane*>(ptr);

        AirPlane* curr = freeStore;
        for (int i = 1; i < maxCount; i++)
        {
            auto p = reinterpret_cast<AirPlane*>(ptr + i * m_size);
            curr->store_.next = p;
            curr->state_ = State::FREE;
            curr = p;
        }
        curr->store_.next = nullptr;
        curr->state_ = State::FREE;
    }
    AirPlane* firstStorage = freeStore;
    freeStore = freeStore->store_.next;
    return firstStorage;
}

void AirPlane::operator delete(void *mem) noexcept
{
    AirPlane* curr = reinterpret_cast<AirPlane*>(mem);
    curr->store_.next = freeStore;
    freeStore = curr;
}

vector<string> wordBreak1(string s, vector<string> &wordDict)
{
    unordered_set<string> dict(wordDict.begin(), wordDict.end());
    
    int n = static_cast<int>(s.size());
    vector<bool> dcp(n + 1, false);
    dcp[0] = true;

    vector<vector<int> > matched_pos(n + 1);
    for (int i = 1; i <= n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            if(dcp[j] && dict.count(s.substr(j, i - j)))
            {
                dcp[i] = true;
                matched_pos[i].push_back(j);
            }
        }
    }
    vector<string> ans;
    if(!dcp[n]) return ans; 

    stack<pair<int, vector<string> > > m_stack;
    m_stack.push(make_pair(n, vector<string>{}));

    while (!m_stack.empty())
    {
        auto pos = m_stack.top().first;
        auto res = m_stack.top().second;
        m_stack.pop();

        if(pos == 0)
        {
            reverse(res.begin(), res.end());
            string one_ans;
            for (int i = 0; i < (int)res.size(); i++)
            {
                if(i) one_ans += " ";
                one_ans += res[i];
            }
            ans.push_back(one_ans);
            continue;
        }

        for (const auto& p : matched_pos[pos])
        {
            vector<string> path = res;
            path.push_back(s.substr(p, pos - p));
            m_stack.push({p, path});
        }
    }
    return ans;
}

bool hasCycle(ListNode *head)
{
    if(head == nullptr || head->next == nullptr) return false;

    ListNode* fast_ptr = head;
    ListNode* slow_ptr = head;
    while (fast_ptr && fast_ptr->next)
    {
        fast_ptr = fast_ptr->next->next;
        slow_ptr = slow_ptr->next;

        if(fast_ptr == slow_ptr)
            return true;
    }
    return false;
}

// vector<VideoInfo> traverse_file_on_direction(const string &src_dir, const string& suffix)
// {
//     vector<VideoInfo> all_videos;
//     deque<string> dirs;
//     dirs.push_back(src_dir);
//     while (!dirs.empty())
//     {
//         string top = dirs.front();
//         dirs.pop_front();

//         DIR* dir = opendir(top.c_str());
//         struct dirent* ent;

//         while((ent = readdir(dir)) != nullptr)
//         {
//             string name  = ent->d_name;
//             if(!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."))
//                 continue;

//             string fullPath = top + "/" + name;
//             struct stat st;
//             if(stat(fullPath.c_str(), &st) == -1)
//             {
//                 cerr << "read error: " << top << "\n";
//                 continue;
//             }

//             if(S_ISDIR(st.st_mode))
//                 dirs.push_back(fullPath);
//             else
//             {
//                 if(strstr(ent->d_name, suffix.c_str()) != nullptr)
//                     all_videos.push_back(VideoInfo{string(name), fullPath, ((double)st.st_size/1024/1024/1024)});
//             }        
//         }
//     }
//     return all_videos;
// }

// void transform_quality(const vector<VideoInfo> &files, const string &dst_dir)
// {
//     for (int i = 0; i < (int)files.size(); i++)
//     {
//         string file_path = dst_dir + "/" + files[i].name;
//         if(files[i].size > 3.5)
//         {
//             copy_file(files[i].path, file_path);
//             continue;
//         }

//         cv::VideoCapture cap(files[i].path.c_str());
//         if(!cap.isOpened())
//         {
//             cerr << files[i].path << "cannot open!!!" << "\n";
//             continue;
//         }

//         int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
//         int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
//         double fps = cap.get(cv::CAP_PROP_FPS);        

//         cv::Size frame_size(width, height);

//         int fourcc = cv::VideoWriter::fourcc('H','2','6','4');
//         cv::VideoWriter writer(
//             file_path.c_str(), 
//             fourcc, 
//             fps, 
//             frame_size
//         );

//         if (!writer.isOpened()) 
//         {
//             cerr << "VideoWriter failed!!!\n";
//             continue;
//         }

//         cv::Mat frame;
//         while (cap.read(frame)) 
//         {
//             writer.write(frame);
//         }
//         cap.release();
//         writer.release();
//         std::cout << "完成！\n";
//     }

// }

// void copy_file(const string &src_path, const string &dst_path)
// {
//     ifstream in(src_path.c_str(), ios::binary);
//     ofstream out(dst_path.c_str(), ios::binary);
//     if(!in || !out) return;

//     out << in.rdbuf();

//     in.close();
//     out.close();
// }

void test()
{
    // ListNode* list_element = createListNode({21, 34, 54, 65, 3, 35, 563}); 
    // reverseKGroup(list_element, 3);
    // removeElement(nums, 2);
    // vector<vector<char> > nums = {{'.','.','5','.','.','.','.','.','.'},
    //                               {'1','.','.','2','.','.','.','.','.'},
    //                               {'.','.','6','.','.','3','.','.','.'},
    //                               {'8','.','.','.','.','.','.','.','.'},
    //                               {'3','.','1','5','2','.','.','.','.'},
    //                               {'.','.','.','.','.','.','.','4','.'},
    //                               {'.','.','6','.','.','.','.','.','.'},
    //                               {'.','.','.','.','.','.','.','9','.'},
    //                               {'.','.','.','.','.','.','.','.','.'}};
    // vector<vector<char> > nums = {{'1','0','1','0','0'},{'1','0','1','1','1'},{'1','1','1','1','1'},{'1','0','0','1','0'}};
    // read_probe_list("probe_list.txt");
    // vector<int> nums = {12,4,35,45,3,54,65,45,564,5,553};
    // heapSort(nums);
    // struct ComStudent //仿函数
    // {
    //     bool operator()(const Student* stu1, const Student* stu2) const
    //     {
    //         if(stu1->name != stu2->name) return stu1->name<stu2->name;
    //         if(stu1->age != stu2->age) return stu1->age < stu2->age;
    //         return stu1->id < stu2->id;
    //     }
    // };

    // map<Student*, int, ComStudent> m_map_pointer;
    // priority_queue<Student*, vector<Student*>, ComStudent> m_queue;
    // Student* stu1_pointer = new Student("ck", 29, 12323);
    // Student* stu2_pointer = new Student("bc", 30, 12323);
    // Student* stu3_pointer = new Student("ck", 35, 12323);
    // Student* stu4_pointer = new Student("ck", 25, 12323);
    // Student* stu5_pointer = new Student("ck", 30, 12323);
    // m_queue.push(stu1_pointer);
    // m_queue.push(stu2_pointer);
    // m_queue.push(stu3_pointer);
    // m_queue.push(stu4_pointer);
    // m_queue.push(stu5_pointer);
    // while (!m_queue.empty())
    // {
    //     cout << m_queue.top()->name << " : " << m_queue.top()->age << "\n";
    //     // m_queue.pop();
    // }
    // stu1_pointer->name = "cd";

    // while (!m_queue.empty())
    // {
    //     cout << m_queue.top()->name << " : " << m_queue.top()->age << "\n";
    //     m_queue.pop();
    // }

    // map<Student, int> m_map;
    // Student stu1("ck", 29, 12323);
    // Student stu2("bc", 30, 12323);
    // Student stu3("ck", 35, 12323);
    // Student stu4("ck", 25, 12323);
    // Student stu5("ck", 30, 12323);
    // m_map.insert(make_pair(stu1, 1));
    // m_map.insert(make_pair(stu2, 1));
    // m_map.insert(make_pair(stu3, 1));
    // m_map.insert(make_pair(stu4, 1));
    // m_map.insert(make_pair(stu5, 1));
    // vector<vector<int> > nums = {{1, 2, 3, 4, 5},
    //                              {6, 7, 8, 9, 10},
    //                              {11,12,13,14,15}};

    // vector<tuple<int, int, double> > datas = {{2,1,0.9},
    //                                           {2,4,5},
    //                                           {5,4,0.5},
    //                                           {3,5,0.8},
    //                                           {1,3,3},
    //                                           {1,6,9},
    //                                           {1,4,2},
    //                                           {5,1,0.1}
    //                                           };
    // unique_ptr<Graph<int> > m_grap(new Graph(datas));

    // // vector<Edge<int> > edges;
    // auto ret = m_grap->minimum_generation_path(1);
    // for (auto iter = ret.begin(); iter != ret.end(); iter++)
    // {
    //     cout << 1 << " to " << iter->first->get_value() << " : " << iter->second << "\n";
    // }
    
    // vector<int> gas  = {1,2,3,4,5};
    // vector<vector<int> > cost = {
    //     {1,2},
    //     {3,5},
    //     {2,2}
    // };
    // const string word = "helloworld";
    // const int N = 64;
    // const long long M = 7;

    // int idx = 0;
    // char buff[N + 1] = { 0 };
    // for (int i = 0; i < M; i++)
    // {
    //     for (int j=0; j<(int)word.size(); j++)
    //     {
    //         buff[idx] = word[j];
    //         ++idx;
    //         if(idx == N) idx = 0; 
    //     }
    // }
    // buff[N] = '\0';
    // printf("%s\n", buff);
    //////////////////////////////////////////////////////////////////////////////////
    // auto pool = MemoryAllocateWay::init_nonfix_pool(2, 129, 100);
    // int* ptr = (int*)MemoryAllocateWay::allocate_nonfixed_pool(pool, sizeof(int));
    // *ptr = 10;
    // MemoryAllocateWay::deallocate_nonfixed_pool(pool, (void*)ptr, sizeof(int));
    // MemoryAllocateWay::free_nonfixed_pool(pool);
    //////////////////////////////////////////////////////////////////////////////////
    // CustMultiSet<int> m_set;
    // m_set.insert(2);
    // m_set.insert(8);
    // m_set.insert(3);
    // m_set.insert(1);
    // list<int, __gnu_cxx::__pool_alloc<int> > custom_lists;
    // for (int i = 0; i < 10000000; i++)
    // {
    //     custom_lists.push_back(i);
    // }
    // cout << "block_size: " << block_size << ", block_time: " << block_time << "\n";
    cout << "hello world" << "\n";
}
