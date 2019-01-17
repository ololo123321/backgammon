#include <random>
#include <iostream>
#include <array>
#include <list>
#include <map>
#include <unordered_set>
#include <algorithm>
using namespace std;

struct node { 
    array<int, 24> board; 
    map<int, int> bar;
    list<int> towers;
    int depth;
};

struct node_comparator {
    bool operator()(const node &lhs, const node &rhs) const {
        if (lhs.board == rhs.board && lhs.bar == rhs.bar && lhs.depth == rhs.depth){
            return true;
        } else {
            return false;
        }
    }
};

struct node_hasher {
    int operator()(const node &n) const {
        string res = "";
        int size = n.board.size();
        for (auto const& i : n.board){
            res += to_string(i);
        }
        for (auto const& kv : n.bar){
            res += to_string(kv.second);
        }
        return hash<string>()(res);
    }
};

node create_node(array<int, 24> board, map<int, int> bar, list<int> towers, int depth){
    node n;
    n.board = board;
    n.bar = bar;
    n.towers = towers;
    n.depth = depth;
    return n;
}

auto roll_dice() {
    random_device rd; 
    mt19937 gen(rd()); 
    uniform_int_distribution<> dis(1, 6);

    int r1 = dis(gen);
    int r2 = dis(gen);
    if (r1 == r2){
        return vector<int> {r1, r1, r1, r1};
    } else {
        return vector<int> {r1, r2};
    }
}

bool is_valid_move(int t, int r, array<int, 24> board, list<int> towers) {
    int end = t + r;
    if (end <= 23){
        if (board[end] >= -1){
            return true;
        } else {
            return false;
        }
    } else {
        int t_min = *min_element(towers.begin(), towers.end());
        if (t_min >= 18 && (r == 24 - t || t == t_min)){
            return true;
        } else {
            return false;
        }
    }
}

void add_piece(array<int, 24>& board, map<int, int>& bar, int p, list<int>& towers){
    if (bar[1] > 0){
        bar[1] -= 1;
    }
    if (board[p] >= 0){
        board[p] += 1;
    }
    if (board[p] == -1){
        board[p] = 1;
        bar[-1] += 1;
    }
    if (board[p] == 1){
        towers.push_back(p);
    }
}

struct State{
    vector<int> roll = roll_dice();
    array<int, 24> board = {2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2};
    map<int, int> bar = {{1, 0}, {-1, 0}};
    int max_depth = roll.size();
    list<int> towers;
    unordered_set<node, node_hasher, node_comparator> leaves;
    int winner = 0;
    int sign = 1;
    
    void extend(node n){
        array<int, 24> board_;
        map<int, int> bar_;
        list<int> towers_;
        node n_new;

        if (winner){
           return;
        }

        if (n.depth == max_depth){
            leaves.insert(n);
            return;
        }

        int r = roll[n.depth];
        if (n.bar[1] != 0){
            if (n.board[r-1] >= -1){
                tie(board_, bar_, towers_) = make_tuple(n.board, n.bar, n.towers);
                add_piece(board_, bar_, r-1, towers_);
                n_new = create_node(board_, bar_, towers_, n.depth+1);
                extend(n_new);
            } else {
                if (max_depth == 2){
                    n_new = n;
                    n_new.depth += 1;
                    extend(n_new);
                } else {
                    leaves.insert(n);
                }
            }
        } else {
            bool is_leaf = true;
            for (auto const& t : n.towers){
                if (is_valid_move(t, r, n.board, n.towers)){
                    is_leaf = false;
                    tie(board_, bar_, towers_) = make_tuple(n.board, n.bar, n.towers);
                    board_[t] -= 1;
                    if (board_[t] == 0){
                        towers_.remove(t);
                    }
                    int end = t + r;
                    if (end <= 23){
                        add_piece(board_, bar_, end, towers_);
                    }
                    if (towers_.size() == 0){
                        winner = sign;
                        leaves.insert(n);
                        return;
                    }
                    n_new = create_node(board_, bar_, towers_, n.depth+1);
                    extend(n_new);
                }
            }
            if (is_leaf){
                leaves.insert(n);
            }
        }
    }

    auto get_moves(){
		list<int> towers;
        for (int i = 0; i < board.size(); ++i){
            if (board[i] > 0){
                towers.push_back(i);
            }
        }
        
        for (const auto& t: towers){
            cout << t << " ";
        }
        
        int depth = 0;  
        node root = create_node(board, bar, towers, depth);
        extend(root);
        if (roll.size() == 2){
            roll = {roll[1], roll[0]};
            extend(root);
        }
        
        int d_max = 0;
        for (const auto& n: leaves) {
            if (n.depth > d_max){
                d_max = n.depth;
            }
        }
        
        list<pair<array<int, 24>, map<int, int>>> res;
        for (const auto& n: leaves) {
            if (n.depth == d_max){
                res.push_back({n.board, n.bar});
            }
        }
        return res;
    }
    
    void update(pair<array<int, 24>, map<int, int>> move){
        board = move.first;
        bar = move.second;
        for (int i = 0; i < board.size(); ++i){
            board[i] *= -1;
        }
        reverse(board.begin(), board.end());
        
        int q = bar[-1];
        bar[-1] = bar[1];
        bar[1] = q;
        
        roll = roll_dice();
        max_depth = roll.size();
        sign *= -1;
    }
	
	void update_randomly(){
        list<pair<array<int, 24>, map<int, int>>> moves = get_moves();

        vector<pair<array<int, 24>, map<int, int>>> out;
        random_device rd; 
        mt19937 g(rd()); 
        sample(moves.begin(), moves.end(), back_inserter(out), 1, g);
        
        board = out[0].first;
        bar = out[0].second;
        for (int i = 0; i < board.size(); ++i){
            board[i] *= -1;
        }
        reverse(board.begin(), board.end());
        
        int q = bar[-1];
        bar[-1] = bar[1];
        bar[1] = q;
        
        roll = roll_dice();
        max_depth = roll.size();
        sign *= -1;
    }
};

void pit(){
	while (true){
		s.update_randomly();
		if (s.winner){
			break;
		}
	}
}
