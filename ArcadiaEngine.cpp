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

//possibilities calculation
const long long MOD = 1e9 + 7;

// =========================================================
// PART A: DATA STRUCTURES (Concrete Implementations)
// =========================================================

// --- 1. PlayerTable (Double Hashing) ---

class ConcretePlayerTable : public PlayerTable {
    private:
    struct Entry {
        int id;
        string name;
        bool occupied = false;
    };

    vector<Entry> table;
    int count = 0;

    public:
    ConcretePlayerTable() {
        table.resize(101); //fixed size of 101
    }
    
    void insert(int playerID, string name) {
        if (count >= 101) {
            cout << "Table is Full" << endl;
            return;
        } // if the table is full 

        //the hashes
        int h1 = playerID % 101;
        int h2 = 97 - (playerID % 97); //to prevent the infinite loop

        int idx = h1;
        int i = 0;

        //searching
        while (table[idx].occupied) {
            i++;
            idx = (h1 + (i * h2)) % 101;
        }

        //inserting
        table[idx].id = playerID;
        table[idx].name = name;
        table[idx].occupied = true;
        count++;
    };

    string search(int playerID) {
        //hashes
        int h1 = playerID % 101;
        int h2 = 97 - (playerID % 97);
        
        int idx = h1;
        int i = 0;

        
        while (i < 101) {
            //if no player
            if (!table[idx].occupied) {
                return "";
            }

            //same id = found
            if (table[idx].id == playerID) {
                return table[idx].name;
            }
            //else jumb and repeat*-9+--
            +
            i++;
            idx = (h1 + (i * h2)) % 101;
        }

        return "";
    }
};

// --- 2. Leaderboard (Skip List) ---

const int MAX_LEVEL = 16; // Maximum level for the Skip List
const float P = 0.5;      // Probability factor for random level generation

class ConcreteLeaderboard : public Leaderboard {
private:
    struct PlayerNode {
        int playerID;
        int score;
       vector<PlayerNode*> forward;

        PlayerNode(int id, int scr, int level) : playerID(id), score(scr) {
            forward.resize(level + 1, nullptr);
        }
    };

    PlayerNode* head;
    int current_level;

    // Helper to generate a random level for a new node
    int randomLevel() {
        int level = 0;
        while (rand() / (RAND_MAX + 1.0) < P && level < MAX_LEVEL) {
            level++;
        }
        return level;
    }

    // helper function for insertion/search
    bool shouldBePlacedBefore(int scoreA, int idA, const PlayerNode* nodeB) const {
        if (scoreA != nodeB->score) {
            return scoreA > nodeB->score; //Descending for the score
        }
        return idA < nodeB->playerID; //Ascending for the id
    }

    // helper function for deletion Returns true if nodeA comes before nodeB in the list
    bool comesBefore(const PlayerNode* nodeA, const PlayerNode* nodeB) const {
        if (nodeA->score != nodeB->score) {
            return nodeA->score > nodeB->score;
        }
        return nodeA->playerID < nodeB->playerID;
    }

public:
    // clean up
    ~ConcreteLeaderboard() {
        PlayerNode* current = head;
        PlayerNode* next_node = nullptr;
        while (current != nullptr) {
            next_node = current->forward[0];
            delete current;
            current = next_node;
        }
    }

    ConcreteLeaderboard() : current_level(0) {
        head = new PlayerNode(-1, -1, MAX_LEVEL);
    }

    void addScore(int playerID, int score) override {
        vector<PlayerNode*> update(MAX_LEVEL + 1);
        PlayerNode* current = head;

        // 1. Find the update array (predecessors) in O(log n)
        for (int i = current_level; i >= 0; i--) {
            // Traverse forward as long as the next node should be placed before the new key
            while (current->forward[i] != nullptr &&
                   shouldBePlacedBefore(score, playerID, current->forward[i])) {
                current = current->forward[i];
            }
            update[i] = current;
        }


        int new_level = randomLevel();

        // Update max level
        if (new_level > current_level) {
            for (int i = current_level + 1; i <= new_level; i++) {
                update[i] = head;
            }
            current_level = new_level;
        }


        PlayerNode* new_node = new PlayerNode(playerID, score, new_level);

        for (int i = 0; i <= new_level; i++) {
            new_node->forward[i] = update[i]->forward[i];
            update[i]->forward[i] = new_node;
        }
    }

    void removePlayer(int playerID) override {
        PlayerNode* node_to_delete = nullptr;

        PlayerNode* runner = head->forward[0];
        while (runner != nullptr) {
            if (runner->playerID == playerID) {
                node_to_delete = runner;
                break;
            }
            runner = runner->forward[0];
        }

        // if the node isn't found
        if (node_to_delete == nullptr) {
            return;
        }


        PlayerNode* current = head;

        // Traverse from the highest level down
        for (int i = current_level; i >= 0; i--) {

            // search for the node comes before node_to_delete   time : O(log n)
            while (current->forward[i] != nullptr &&
                   comesBefore(current->forward[i], node_to_delete)) {
                current = current->forward[i];
            }

            // if current->forward[i] is the node we found update it to the forward of the node_to_delete
            if (current->forward[i] == node_to_delete) {
                current->forward[i] = node_to_delete->forward[i];
            }
        }

        // Clean up list level
        while (current_level > 0 && head->forward[current_level] == nullptr) {
            current_level--;
        }

        delete node_to_delete;
    }

    vector<int> getTopN(int n) override {
        std::vector<int> top_n_players;
        // Start at the first node
        PlayerNode* current = head->forward[0];

        // store the players until size n
        while (current != nullptr && top_n_players.size() < n) {
            top_n_players.push_back(current->playerID);
            current = current->forward[0];
        }

        return top_n_players;
    }
};

// --- 3. AuctionTree (Red-Black Tree) ---

class ConcreteAuctionTree : public AuctionTree {
private:
    // TODO: Define your Red-Black Tree node structure
    // Hint: Each node needs: id, price, color, left, right, parent pointers

public:
    ConcreteAuctionTree() {
        // TODO: Initialize your Red-Black Tree
    }

    void insertItem(int itemID, int price) override {
        // TODO: Implement Red-Black Tree insertion
        // Remember to maintain RB-Tree properties with rotations and recoloring
    }

    void deleteItem(int itemID) override {
        // TODO: Implement Red-Black Tree deletion
        // This is complex - handle all cases carefully
    }
};

// =========================================================
// PART B: INVENTORY SYSTEM (Dynamic Programming)
// =========================================================

int InventorySystem::optimizeLootSplit(int n, vector<int>& coins) {
    if (coins.empty()) {
        return 0;
    }

    for (int coin : coins) {
        if (coin < 0) {
            cerr << "Error: Coin values must be non-negative." << endl;
            return -1; 
        }
    }

    long long totalSumLL = 0;
    for (int coin : coins) {
        totalSumLL += coin;
    }

    if (totalSumLL > INT_MAX || totalSumLL > 500000) { 
        cerr << "Warning: Total coin sum too large for standard DP array. Truncating target." << endl;
    }
    
    int totalSum = (int)min((long long)INT_MAX, totalSumLL);
    int target = totalSum / 2;

    // DP[i] = true if a subset sum of 'i' is possible.
    vector<bool> dp(target + 1, false);
    dp[0] = true; 

    // Use 0/1 Subset Sum DP
    for (int coin : coins) {
        // Only process coins that fit within the safe target sum
        if (coin <= target) { 
            for (int j = target; j >= coin; --j) {
                dp[j] = dp[j] || dp[j - coin];
            }
        }
    }

    // Find the largest possible S_subset <= target
    int sSubset = 0;
    for (int i = target; i >= 0; --i) {
        if (dp[i]) {
            sSubset = i;
            break;
        }
    }

    // Recalculate difference using the safer long long total sum
    return (int)(totalSumLL - 2 * (long long)sSubset);
}


int InventorySystem::maximizeCarryValue(int capacity, vector<pair<int, int>>& items) {
    if (capacity <= 0 || items.empty()) {
        return 0;
    }
    
    // Input Validation: Check for invalid (negative) weight/value
    for (const auto& item : items) {
        if (item.first < 0 || item.second < 0) {
            cerr << "Error: Item weight and value must be non-negative." << endl;
            return -1;
        }
    }

    // DP[w]: Max value for capacity 'w'.
    // If capacity or total value can exceed INT_MAX, this should be vector<long long>
    vector<int> dp(capacity + 1, 0);

    // Iterate through each item and update DP table
    for (const auto& item : items) {
        int weight = item.first;
        int value = item.second;

        // Iterate backwards (0/1 Knapsack)
        for (int w = capacity; w >= weight; --w) {
            dp[w] = max(dp[w], dp[w - weight] + value);
        }
    }

    return dp[capacity];
}


long long InventorySystem::countStringPossibilities(string s) {
    int len = s.length();
    if (len == 0) {
        return 1;
    }

    // Input Validation: Check for characters that cannot be part of the RECEIVED string.
    for (char c : s) {
        if (c == 'w' || c == 'm') {
            return 0;
        }
        if (!std::isalpha(c) && !std::isdigit(c) && c != 'u' && c != 'n') {
            cerr << "Error: Invalid character '" << c << "' in input string." << endl;
            return 0; 
        }
    }

    // DP[i]: Number of ways to decode the prefix s[0...i-1] (length i)
    vector<long long> dp(len + 1, 0);
    dp[0] = 1; 

    for (int i = 1; i <= len; ++i) {
        // Decode s[i-1] as a single character (Always possible)
        dp[i] = dp[i-1];

        // Decode s[i-2]s[i-1] as a single block ('w' or 'm')
        if (i >= 2) {
            string pair = s.substr(i - 2, 2);

            if (pair == "uu" || pair == "nn") {
                // This adds the possibility of 'w' or 'm' decoding, extending dp[i-2] ways.
                dp[i] = (dp[i] + dp[i-2]) % MOD;
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
    // TODO: Implement path existence check using BFS or DFS
    // edges are bidirectional
    return false;
}

long long WorldNavigator::minBribeCost(int n, int m, long long goldRate, long long silverRate,
                                       vector<vector<int>>& roadData) {
    // TODO: Implement Minimum Spanning Tree (Kruskal's or Prim's)
    // roadData[i] = {u, v, goldCost, silverCost}
    // Total cost = goldCost * goldRate + silverCost * silverRate
    // Return -1 if graph cannot be fully connected
    return -1;
}

string WorldNavigator::sumMinDistancesBinary(int n, vector<vector<int>>& roads) {
    // TODO: Implement All-Pairs Shortest Path (Floyd-Warshall)
    // Sum all shortest distances between unique pairs (i < j)
    // Return the sum as a binary string
    // Hint: Handle large numbers carefully
    return "0";
}

// =========================================================
// PART D: SERVER KERNEL (Greedy)
// =========================================================


int ServerKernel::minIntervals(vector<char>& tasks, int n) {
    int counts[26] = {0}; //letters
    int max_repeats = 0;


    for (char t : tasks) {
    
        int index = t - 'A'; // a trick to convert the letter to numbers .. A = 65
        
        //adding 1 for every occurence of this letter in counts
        counts[index]++;
        
        // setting the max
        if (counts[index] > max_repeats) {
            max_repeats = counts[index];
        }
    }

    int tasks_with_max_repeats = 0;

    for (int c : counts) {
        if (c == max_repeats) {
            tasks_with_max_repeats++;
        }
    }

    int number_of_full_rows = max_repeats - 1; //max repeat govern what would the grid look like - the last one
    int size_of_row = n + 1; // the max repeat + waiting time

    int result = (number_of_full_rows * size_of_row) + tasks_with_max_repeats;
    return max(result, (int)tasks.size()); //you have to wait for the tasks if no repeating
}


// =========================================================
// FACTORY FUNCTIONS (Required for Testing)
// =========================================================

extern "C" {
    PlayerTable* createPlayerTable() { 
        return new ConcretePlayerTable(); 
    }

    Leaderboard* createLeaderboard() { 
        return new ConcreteLeaderboard(); 
    }

    AuctionTree* createAuctionTree() { 
        return new ConcreteAuctionTree(); 
    }
}
