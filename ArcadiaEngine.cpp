// ArcadiaEngine.cpp - STUDENT TEMPLATE
// TODO: Implement all the functions below according to the assignment requirements

#include "ArcadiaEngine.h"
#include <algorithm>
#include <queue>
#include <numeric>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <set>

using namespace std;

// possibilities calculation
const long long MOD = 1e9 + 7;

// =========================================================
// PART A: DATA STRUCTURES (Concrete Implementations)
// =========================================================

// --- 1. PlayerTable (Double Hashing) ---

class ConcretePlayerTable : public PlayerTable
{
private:
    struct Entry
    {
        int id;
        string name;
        bool occupied = false;
    };

    vector<Entry> table;
    int count = 0;

public:
    ConcretePlayerTable()
    {
        table.resize(101); // fixed size of 101
    }

    void insert(int playerID, string name)
    {
        if (count >= 101)
        {
            cout << "Table is Full" << endl;
            return;
        } // if the table is full

        // the hashes
        int h1 = playerID % 101;
        int h2 = 97 - (playerID % 97); // to prevent the infinite loop

        int idx = h1;
        int i = 0;

        // searching
        while (table[idx].occupied)
        {
            i++;
            idx = (h1 + (i * h2)) % 101;
        }

        // inserting
        table[idx].id = playerID;
        table[idx].name = name;
        table[idx].occupied = true;
        count++;
    };

    string search(int playerID)
    {
        // hashes
        int h1 = playerID % 101;
        int h2 = 97 - (playerID % 97);

        int idx = h1;
        int i = 0;

        while (i < 101)
        {
            // if no player
            if (!table[idx].occupied)
            {
                return "";
            }

            // same id = found
            if (table[idx].id == playerID)
            {
                return table[idx].name;
            }
            // else jumb and repeat*-9+--
            +i++;
            idx = (h1 + (i * h2)) % 101;
        }

        return "";
    }
};

// --- 2. Leaderboard (Skip List) ---

const int MAX_LEVEL = 16; // Maximum level for the Skip List
const float P = 0.5;      // Probability factor for random level generation

class ConcreteLeaderboard : public Leaderboard
{
private:
    struct PlayerNode
    {
        int playerID;
        int score;
        vector<PlayerNode *> forward;

        PlayerNode(int id, int scr, int level) : playerID(id), score(scr)
        {
            forward.resize(level + 1, nullptr);
        }
    };

    PlayerNode *head;
    int current_level;

    // Helper to generate a random level for a new node
    int randomLevel()
    {
        int level = 0;
        while (rand() / (RAND_MAX + 1.0) < P && level < MAX_LEVEL)
        {
            level++;
        }
        return level;
    }

    // helper function for insertion/search
    bool shouldBePlacedBefore(int scoreA, int idA, const PlayerNode *nodeB) const
    {
        if (scoreA != nodeB->score)
        {
            return scoreA > nodeB->score; // Descending for the score
        }
        return idA < nodeB->playerID; // Ascending for the id
    }

    // helper function for deletion Returns true if nodeA comes before nodeB in the list
    bool comesBefore(const PlayerNode *nodeA, const PlayerNode *nodeB) const
    {
        if (nodeA->score != nodeB->score)
        {
            return nodeA->score > nodeB->score;
        }
        return nodeA->playerID < nodeB->playerID;
    }

public:
    // clean up
    ~ConcreteLeaderboard()
    {
        PlayerNode *current = head;
        PlayerNode *next_node = nullptr;
        while (current != nullptr)
        {
            next_node = current->forward[0];
            delete current;
            current = next_node;
        }
    }

    ConcreteLeaderboard() : current_level(0)
    {
        head = new PlayerNode(-1, -1, MAX_LEVEL);
    }

    void addScore(int playerID, int score) override
    {
        vector<PlayerNode *> update(MAX_LEVEL + 1);
        PlayerNode *current = head;

        // 1. Find the update array (predecessors) in O(log n)
        for (int i = current_level; i >= 0; i--)
        {
            // Traverse forward as long as the next node should be placed before the new key
            while (current->forward[i] != nullptr &&
                   shouldBePlacedBefore(score, playerID, current->forward[i]))
            {
                current = current->forward[i];
            }
            update[i] = current;
        }

        int new_level = randomLevel();

        // Update max level
        if (new_level > current_level)
        {
            for (int i = current_level + 1; i <= new_level; i++)
            {
                update[i] = head;
            }
            current_level = new_level;
        }

        PlayerNode *new_node = new PlayerNode(playerID, score, new_level);

        for (int i = 0; i <= new_level; i++)
        {
            new_node->forward[i] = update[i]->forward[i];
            update[i]->forward[i] = new_node;
        }
    }

    void removePlayer(int playerID) override
    {
        PlayerNode *node_to_delete = nullptr;

        PlayerNode *runner = head->forward[0];
        while (runner != nullptr)
        {
            if (runner->playerID == playerID)
            {
                node_to_delete = runner;
                break;
            }
            runner = runner->forward[0];
        }

        // if the node isn't found
        if (node_to_delete == nullptr)
        {
            return;
        }

        PlayerNode *current = head;

        // Traverse from the highest level down
        for (int i = current_level; i >= 0; i--)
        {

            // search for the node comes before node_to_delete   time : O(log n)
            while (current->forward[i] != nullptr &&
                   comesBefore(current->forward[i], node_to_delete))
            {
                current = current->forward[i];
            }

            // if current->forward[i] is the node we found update it to the forward of the node_to_delete
            if (current->forward[i] == node_to_delete)
            {
                current->forward[i] = node_to_delete->forward[i];
            }
        }

        // Clean up list level
        while (current_level > 0 && head->forward[current_level] == nullptr)
        {
            current_level--;
        }

        delete node_to_delete;
    }

    vector<int> getTopN(int n) override
    {
        std::vector<int> top_n_players;
        // Start at the first node
        PlayerNode *current = head->forward[0];

        // store the players until size n
        while (current != nullptr && top_n_players.size() < n)
        {
            top_n_players.push_back(current->playerID);
            current = current->forward[0];
        }

        return top_n_players;
    }
};

// --- 3. AuctionTree (Red-Black Tree) ---

class ConcreteAuctionTree : public AuctionTree {
private:
    enum Color { RED, BLACK };

    struct Node {
        int id;
        int price;
        Color color;
        Node* left;
        Node* right;
        Node* parent;

        Node(int itemID, int itemPrice) : id(itemID), price(itemPrice), color(RED),
                                          left(nullptr), right(nullptr), parent(nullptr) {}
    };

    Node* root;

    // Compare nodes by price, then by id
    bool lessThan(Node* a, Node* b) const {
        return (a->price < b->price) ||
               (a->price == b->price && a->id < b->id);
    }
    // Find node by id using iterative inorder traversal
    Node* findByID(int itemID) {
        if (root == nullptr) return nullptr;

        vector<Node*> stack;
        Node* current = root;

        while (current != nullptr || !stack.empty()) {
            while (current != nullptr) {
                stack.push_back(current);
                current = current->left;
            }
            current = stack.back();
            stack.pop_back();

            if (current->id == itemID) return current;
            current = current->right;
        }
        return nullptr;
    }


public:
    ConcreteAuctionTree() : root(nullptr) {}
    ~ConcreteAuctionTree() { clear(root); }

    // BST insertion with fix for RBT properties
    void insertItem(int itemID, int price) override {
        Node* z = new Node(itemID, price);
        Node* y = nullptr;
        Node* x = root;

        while (x != nullptr) {
            y = x;
            if (lessThan(z, x))
                x = x->left;
            else
                x = x->right;
        }

        z->parent = y;
        if (y == nullptr)
            root = z;
        else if (lessThan(z, y))
            y->left = z;
        else
            y->right = z;

        fixInsert(z);
    }

    void deleteItem(int itemID) override {
        Node* z = findByID(itemID);
        if (z == nullptr) return;
        deleteNode(z);
    }

private:

    // delete all nodes
    void clear(Node* node) {
        if (node == nullptr) return;
        clear(node->left);
        clear(node->right);
        delete node;
    }
    // left most node in subtree
    Node* minimum(Node* node) {
        while (node != nullptr && node->left != nullptr)
            node = node->left;
        return node;
    }

    // replace u with v in tree
    void transplant(Node* u, Node* v) {
        if (u->parent == nullptr)
            root = v;
        else if (u == u->parent->left)
            u->parent->left = v;
        else
            u->parent->right = v;
        if (v != nullptr)
            v->parent = u->parent;
    }

    // BST deletion with fix delete for RBT properties
    void deleteNode(Node* z) {
        Node* y = z;
        Node* x;
        Color y_original_color = y->color;

        if (z->left == nullptr) {
            x = z->right;
            transplant(z, z->right);
        } else if (z->right == nullptr) {
            x = z->left;
            transplant(z, z->left);
        } else {
            y = minimum(z->right);
            y_original_color = y->color;
            x = y->right;
            if (y->parent != z) {
                transplant(y, y->right);
                y->right = z->right;
                y->right->parent = y;
            }
            y->left = z->left;
            y->left->parent = y;
            y->color = z->color;
            transplant(z, y);
        }

        if (y_original_color == BLACK && x != nullptr)
            fixDelete(x);

        delete z;
    }

    // RBT left rotation
    void rotateLeft(Node* x) {
        if (x == nullptr || x->right == nullptr) return;
        Node* y = x->right;
        x->right = y->left;
        if (y->left != nullptr)
            y->left->parent = x;
        y->parent = x->parent;
        if (x->parent == nullptr)
            root = y;
        else if (x == x->parent->left)
            x->parent->left = y;
        else
            x->parent->right = y;
        y->left = x;
        x->parent = y;
    }

    // RBT right rotation
    void rotateRight(Node* x) {
        if (x == nullptr || x->left == nullptr) return;
        Node* y = x->left;
        x->left = y->right;
        if (y->right != nullptr)
            y->right->parent = x;
        y->parent = x->parent;
        if (x->parent == nullptr)
            root = y;
        else if (x == x->parent->right)
            x->parent->right = y;
        else
            x->parent->left = y;
        y->right = x;
        x->parent = y;
    }

    // fix tree after insertion
    void fixInsert(Node* z) {
        while (z != nullptr && z->parent != nullptr && z->parent->color == RED) {
            // Case 1: Parent is left child of grandparent
            if (z->parent == z->parent->parent->left) {
                Node* y = z->parent->parent->right;
                if (y != nullptr && y->color == RED) {
                    z->parent->color = BLACK;  // Case 1a: Uncle is Red: recolor
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                } else {
                    // Case 1b: Uncle is Black: rotations needed
                    if (z == z->parent->right) {
                        z = z->parent;  // Left-Right case: rotate left on parent
                        rotateLeft(z);
                    }
                    // Left-Left case: rotate right on grandparent
                    if (z->parent != nullptr) {
                        z->parent->color = BLACK;
                        if (z->parent->parent != nullptr) {
                            z->parent->parent->color = RED;
                            rotateRight(z->parent->parent);
                        }
                    }
                }
            } else {
                // Case 2: Parent is right child of grandparent
                Node* y = z->parent->parent->left;
                if (y != nullptr && y->color == RED) {
                    z->parent->color = BLACK;  // Case 2a: Uncle is Red: recolor
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                } else {
                    // Case 2b: Uncle is Black: rotations needed
                    if (z == z->parent->left) {
                        z = z->parent;  // Right-Left case: rotate right on parent
                        rotateRight(z);
                    }
                    // Right-Right case: rotate left on grandparent
                    if (z->parent != nullptr) {
                        z->parent->color = BLACK;
                        if (z->parent->parent != nullptr) {
                            z->parent->parent->color = RED;
                            rotateLeft(z->parent->parent);
                        }
                    }
                }
            }
        }
        if (root != nullptr) root->color = BLACK; // root must always be black
    }

    // fix tree after deletion
    void fixDelete(Node* x) {
        while (x != nullptr && x != root && x->color == BLACK) {
            if (x == x->parent->left) {
                Node* w = x->parent->right; // sibling
                if (w != nullptr && w->color == RED) {
                    // Case 1: Sibling is Red: rotate left and recolor
                    w->color = BLACK;
                    x->parent->color = RED;
                    rotateLeft(x->parent);
                    w = x->parent->right;
                }
                if (w != nullptr &&
                    (w->left == nullptr || w->left->color == BLACK) &&
                    (w->right == nullptr || w->right->color == BLACK)) {
                    // Case 2: Sibling's children are Black: recolor sibling
                    w->color = RED;
                    x = x->parent;
                } else {
                    if (w != nullptr && w->right != nullptr && w->right->color == BLACK) {
                        // Case 3: Sibling's right child is Black: rotate right on sibling
                        if (w->left != nullptr) w->left->color = BLACK;
                        w->color = RED;
                        rotateRight(w);
                        w = x->parent->right;
                    }
                    // Case 4: Sibling's right child is Red: rotate left on parent and recolor
                    if (w != nullptr) {
                        w->color = x->parent->color;
                        x->parent->color = BLACK;
                        if (w->right != nullptr) w->right->color = BLACK;
                        rotateLeft(x->parent);
                    }
                    x = root;
                }
            } else {
                // Mirror cases for x being right child
                Node* w = x->parent->left;
                if (w != nullptr && w->color == RED) {
                    // Case 1: sibling Red: rotate right
                    w->color = BLACK;
                    x->parent->color = RED;
                    rotateRight(x->parent);
                    w = x->parent->left;
                }
                if (w != nullptr &&
                    (w->right == nullptr || w->right->color == BLACK) &&
                    (w->left == nullptr || w->left->color == BLACK)) {
                    // Case 2: sibling's children Black: recolor sibling
                    w->color = RED;
                    x = x->parent;
                } else {
                    if (w != nullptr && w->left != nullptr && w->left->color == BLACK) {
                        // Case 3: sibling's left child Black: rotate left
                        if (w->right != nullptr) w->right->color = BLACK;
                        w->color = RED;
                        rotateLeft(w);
                        w = x->parent->left;
                    }
                    // Case 4: sibling's left child Red: rotate right and recolor
                    if (w != nullptr) {
                        w->color = x->parent->color;
                        x->parent->color = BLACK;
                        if (w->left != nullptr) w->left->color = BLACK;
                        rotateRight(x->parent);
                    }
                    x = root;
                }
            }
        }
        if (x != nullptr) x->color = BLACK; // always ensure x is black
    }
};

// =========================================================
// PART B: INVENTORY SYSTEM (Dynamic Programming)
// =========================================================

int InventorySystem::optimizeLootSplit(int n, vector<int> &coins)
{
    if (coins.empty())
    {
        return 0;
    }

    for (int coin : coins)
    {
        if (coin < 0)
        {
            cerr << "Error: Coin values must be non-negative." << endl;
            return -1;
        }
    }

    long long totalSumLL = 0;
    for (int coin : coins)
    {
        totalSumLL += coin;
    }

    if (totalSumLL > INT_MAX || totalSumLL > 500000)
    {
        cerr << "Warning: Total coin sum too large for standard DP array. Truncating target." << endl;
    }

    int totalSum = (int)min((long long)INT_MAX, totalSumLL);
    int target = totalSum / 2;

    // DP[i] = true if a subset sum of 'i' is possible.
    vector<bool> dp(target + 1, false);
    dp[0] = true;

    // Use 0/1 Subset Sum DP
    for (int coin : coins)
    {
        // Only process coins that fit within the safe target sum
        if (coin <= target)
        {
            for (int j = target; j >= coin; --j)
            {
                dp[j] = dp[j] || dp[j - coin];
            }
        }
    }

    // Find the largest possible S_subset <= target
    int sSubset = 0;
    for (int i = target; i >= 0; --i)
    {
        if (dp[i])
        {
            sSubset = i;
            break;
        }
    }

    // Recalculate difference using the safer long long total sum
    return (int)(totalSumLL - 2 * (long long)sSubset);
}

int InventorySystem::maximizeCarryValue(int capacity, vector<pair<int, int>> &items)
{
    if (capacity <= 0 || items.empty())
    {
        return 0;
    }

    // Input Validation: Check for invalid (negative) weight/value
    for (const auto &item : items)
    {
        if (item.first < 0 || item.second < 0)
        {
            cerr << "Error: Item weight and value must be non-negative." << endl;
            return -1;
        }
    }

    // DP[w]: Max value for capacity 'w'.
    // If capacity or total value can exceed INT_MAX, this should be vector<long long>
    vector<int> dp(capacity + 1, 0);

    // Iterate through each item and update DP table
    for (const auto &item : items)
    {
        int weight = item.first;
        int value = item.second;

        // Iterate backwards (0/1 Knapsack)
        for (int w = capacity; w >= weight; --w)
        {
            dp[w] = max(dp[w], dp[w - weight] + value);
        }
    }

    return dp[capacity];
}

long long InventorySystem::countStringPossibilities(string s)
{
    int len = s.length();
    if (len == 0)
    {
        return 1;
    }

    // Input Validation: Check for characters that cannot be part of the RECEIVED string.
    for (char c : s)
    {
        if (c == 'w' || c == 'm')
        {
            return 0;
        }
        if (!std::isalpha(c) && !std::isdigit(c) && c != 'u' && c != 'n')
        {
            cerr << "Error: Invalid character '" << c << "' in input string." << endl;
            return 0;
        }
    }

    // DP[i]: Number of ways to decode the prefix s[0...i-1] (length i)
    vector<long long> dp(len + 1, 0);
    dp[0] = 1;

    for (int i = 1; i <= len; ++i)
    {
        // Decode s[i-1] as a single character (Always possible)
        dp[i] = dp[i - 1];

        // Decode s[i-2]s[i-1] as a single block ('w' or 'm')
        if (i >= 2)
        {
            string pair = s.substr(i - 2, 2);

            if (pair == "uu" || pair == "nn")
            {
                // This adds the possibility of 'w' or 'm' decoding, extending dp[i-2] ways.
                dp[i] = (dp[i] + dp[i - 2]) % MOD;
            }
        }

        dp[i] = dp[i] % MOD;
    }

    return dp[len];
}
// =========================================================
// PART C: WORLD NAVIGATOR (Graphs)
// =========================================================

bool WorldNavigator::pathExists(int n, vector<vector<int>>& edges, int source, int dest) {
    vector<vector<int>> adjacent(n);   // adjacency list
    for (auto& e : edges) {
        adjacent[e[0]].push_back(e[1]);   // add neighbor (bidirectional)
        adjacent[e[1]].push_back(e[0]);
    }

    vector<bool> visited(n, false);   // track visited cities
    queue<int> q;    // BFS queue
    q.push(source);
    visited[source] = true;

    while (!q.empty()) {
        int u = q.front(); q.pop();
        if (u == dest) return true;   // found destination

        for (int v : adjacent[u])
            if (!visited[v]) {
                visited[v] = true;
                q.push(v);   // visit neighbors
            }
    }

    return false;   // destination not reached
}

// Disjoint Set Union (DSU) structure to keep track of connected cities
struct DSU {
    vector<int> parent;  // leader of city i
    vector<int> rank;    // tree height for optimization

    DSU(int n) {
        parent.resize(n);
        rank.assign(n, 0);
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    // Find the leader of a city
    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }

    // Unite two cities. Return true if they were separate
    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;        // already connected
        if (rank[a] < rank[b]) swap(a, b); // attach smaller tree under larger
        parent[b] = a;                   // merge sets
        if (rank[a] == rank[b]) rank[a]++; // update rank if equal
        return true;
    }
};

long long WorldNavigator::minBribeCost(int n, int m, long long goldRate, long long silverRate, vector<vector<int>>& roadData) {

    vector<pair<long long, pair<int, int>>> edges; // store edges as (cost, (city1, city2))

    // calculate cost of each road
    for (auto& r : roadData) {
        long long cost = r[2] * goldRate + r[3] * silverRate; // total cost = gold*rate + silver*rate
        edges.push_back({cost, {r[0], r[1]}});
    }

    // sort edges by cost (cheapest first)
    sort(edges.begin(), edges.end());

    DSU dsu(n);
    long long totalMinCost = 0;     // total cost of selected roads
    int used = 0;     // count of edges used


    for (auto& e : edges) {
        if (dsu.unite(e.second.first, e.second.second)) { // if cities are not connected
            totalMinCost += e.first; // add cost
            used++;          // count the edge
        }
    }

    // check if all cities are connected
    if (used != n - 1) return -1;

    return totalMinCost;
}

string WorldNavigator::sumMinDistancesBinary(int n, vector<vector<int>>& roads) {
    const long long INF = 1e18; // A very large number to represent

    // Initialize distance matrix: distance from i to j
    vector<vector<long long>> dist(n, vector<long long>(n, INF));

    // Distance from a city to itself is 0
    for (int i = 0; i < n; i++)
        dist[i][i] = 0;

    // Set direct road distances
    for (auto& r : roads) {
        dist[r[0]][r[1]] = min(dist[r[0]][r[1]], (long long)r[2]); // handle multiple roads
    }

    // compute all pairs shortest paths
    for (int k = 0; k < n; k++)              // intermediate city
        for (int i = 0; i < n; i++)          // start city
            for (int j = 0; j < n; j++)      // end city
                if (dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];

    // Sum all shortest distances
    long long sum = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (dist[i][j] != INF)         // only consider reachable cities
                sum += dist[i][j];

    // If sum is 0
    if (sum == 0) return "0";

    // Convert sum to binary string
    string binary;
    while (sum > 0) {
        binary = char('0' + (sum % 2)) + binary; // append the least significant bit to front
        sum /= 2;      // divide by 2
    }

    return binary;
}

// =========================================================
// PART D: SERVER KERNEL (Greedy)
// =========================================================

int ServerKernel::minIntervals(vector<char> &tasks, int n)
{
    int counts[26] = {0}; // letters
    int max_repeats = 0;

    for (char t : tasks)
    {

        int index = t - 'A'; // a trick to convert the letter to numbers .. A = 65

        // adding 1 for every occurence of this letter in counts
        counts[index]++;

        // setting the max
        if (counts[index] > max_repeats)
        {
            max_repeats = counts[index];
        }
    }

    int tasks_with_max_repeats = 0;

    for (int c : counts)
    {
        if (c == max_repeats)
        {
            tasks_with_max_repeats++;
        }
    }

    int number_of_full_rows = max_repeats - 1; // max repeat govern what would the grid look like - the last one
    int size_of_row = n + 1;                   // the max repeat + waiting time

    int result = (number_of_full_rows * size_of_row) + tasks_with_max_repeats;
    return max(result, (int)tasks.size()); // you have to wait for the tasks if no repeating
}

// =========================================================
// FACTORY FUNCTIONS (Required for Testing)
// =========================================================

extern "C"
{
PlayerTable *createPlayerTable()
{
    return new ConcretePlayerTable();
}

Leaderboard *createLeaderboard()
{
    return new ConcreteLeaderboard();
}

AuctionTree *createAuctionTree()
{
    return new ConcreteAuctionTree();
}
}
