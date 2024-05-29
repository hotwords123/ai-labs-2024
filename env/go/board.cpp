#include "board.h"

#include <cstring>
#include <vector>
#include <algorithm>

const int INF = 1e9;
const int WHITE = -1, BLACK = 1, EMPTY = 0, DRAW = 2;

const coord<int> directions[4] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

c_GoBoard::c_GoBoard(const int &n):
    n(n),
    board(n, n),
    ko_pos(-INF, -INF),
    consecutive_pass(0),
    step_cnt(0),
    last_step_pos(-INF, -INF),
    last_step_color(-INF)
{
    if (n > MAX_N){
        std::cerr << "Board size(" << n << ") too large! (max:" << MAX_N << ") " << std::endl;
        throw std::exception();
    }
}

const std::vector <int> & c_GoBoard::get_board() const{
    return board.get_data();
}

void c_GoBoard::set_board(const std::vector <int> &board_data){
    if (board_data.size() != (unsigned int) n * n){
        std::cerr << "[c_GoBoard::set_board] invalid board data size: " << board_data.size() << " (expect: " << n * n << ")" << std::endl;
        throw std::exception();
    }
    std::memcpy(board.get_data_ptr(), board_data.data(), sizeof(int) * n * n);
    ko_pos = coord<int>(-INF, -INF);
    consecutive_pass = 0;
    step_cnt = 0;
    last_step_pos = coord<int>(-INF, -INF);
    last_step_color = -INF;
}

int c_GoBoard::count_stone(const int &color){
    int cnt = 0;
    for(int i=0; i<this->n; i++)
        for(int j=0; j<this->n; j++)
            if(this->board[i][j] == color)
                cnt++;
    return cnt;
}

void c_GoBoard::pass_stone(const int &color){
    step_cnt += 1;
    consecutive_pass += 1;
    ko_pos = coord<int>(-INF, -INF);
}

class BreadthFirstSearch {
public:
    BreadthFirstSearch(const array2d<int> &board)
        : board(board), visited(board.get_n(), board.get_m()) {}

    bool is_visited(const coord<int> &pos) const {
        return visited[pos];
    }

    /**
     * Perform a breadth-first search on the board.
     * @param start The starting position of the BFS.
     * @param predicate A function that takes a position as input and returns
     *                 whether the position should be visited.
     * @return A list of positions visited during the BFS.
     */
    template <typename Pred>
    std::vector<coord<int>> search(const coord<int> &start, Pred predicate) {
        std::vector<coord<int>> q;

        q.push_back(start);
        visited[start] = true;

        for (size_t head = 0; head < q.size(); head++) {
            coord<int> pos = q[head];
            for (auto direction : directions) {
                coord<int> neighbor = pos + direction;
                if (!board.is_valid_pos(neighbor) || visited[neighbor]) {
                    continue;
                }
                if (predicate(neighbor)) {
                    q.push_back(neighbor);
                    visited[neighbor] = true;
                }
            }
        }

        return q;
    }

    /**
     * Check if a group of stones will be captured if a stone is added at the
     * specified position.
     * @param victim The position of the stone to be captured.
     * @param source The position of the stone to be added.
     * @return A pair of two values:
     *         - A list of positions of the stones to be captured.
     *         - Whether the group of stones will be captured.
     */
    std::pair<std::vector<coord<int>>, int> capture(const coord<int> &victim, const coord<int> &source) {
        int color = board[victim];
        bool captured = true;
        auto stones = search(victim, [&](const coord<int> &neighbor) {
            if (neighbor == source) {
                return false;
            }
            int neighbor_color = board[neighbor];
            if (neighbor_color == EMPTY) {
                captured = false;
            }
            return neighbor_color == color;
        });
        return std::make_pair(std::move(stones), captured);
    }

    /**
     * Determine the belonging of a group of empty points.
     * @param pos The position of the empty point.
     * @return A pair of two values:
     *         - A list of positions of the empty points.
     *         - The color of the group of empty points.
     */
    std::pair<std::vector<coord<int>>, int> belong(const coord<int> &pos) {
        bool black = false;
        bool white = false;
        auto stones = search(pos, [&](const coord<int> &neighbor) {
            int color = board[neighbor];
            if (color == BLACK) {
                black = true;
            }
            if (color == WHITE) {
                white = true;
            }
            return color == EMPTY;
        });

        int color = EMPTY;
        if (black && !white) {
            color = BLACK;
        }
        if (!black && white) {
            color = WHITE;
        }
        return std::make_pair(std::move(stones), color);
    }

private:
    const array2d<int> &board;
    array2d<char> visited;
};

bool c_GoBoard::add_stone(const coord<int> &pos, const int &color){
    // Add a stone at pos, throw exception if illegal
    // @param pos: position to add stone
    // @param color: color of the stone
    // @return: always true 
    // NOTE: this function not only add a stone, but also remove dead stones
    if (!board.is_valid_pos(pos)){
        std::cerr << "[c_GoBoard::add_stone] invalid pos " << pos << std::endl;
        throw std::exception();
    }
    if (board[pos] != EMPTY){
        std::cerr << "[c_GoBoard::add_stone] position not empty: pos=" << pos
                  << ", color here=" << board[pos] << std::endl;
        throw std::exception();
    }
    if (pos == ko_pos){
        std::cerr << "[c_GoBoard::add_stone] ko rule violation: pos=" << pos << std::endl;
        throw std::exception();
    }

    BreadthFirstSearch bfs(board);

    std::vector<coord<int>> captured_stones;
    bool legal = false;   // Whether this move is legal
    bool maybe_ko = true; // Whether the next move may be a ko

    for (auto direction : directions) {
        coord<int> neighbor = pos + direction;
        if (!board.is_valid_pos(neighbor) || bfs.is_visited(neighbor)) {
            continue;
        }

        int neighbor_color = board[neighbor];
        if (neighbor_color == EMPTY) {
            // The stone has liberties, this move is legal
            legal = true;
        } else {
            // Non-empty neighbor, check if the group has liberties
            auto [stones, captured] = bfs.capture(neighbor, pos);
            if (neighbor_color == color && !captured) {
                // Our stone has liberties, this move is legal
                legal = true;
            }
            if (neighbor_color == -color && captured) {
                // Opponent's stones have no liberties, remove them
                legal = true;
                captured_stones.insert(captured_stones.end(), stones.begin(), stones.end());
            }
        }

        // In the case of a ko, the stone must be surrounded by opponent's stones
        if (neighbor_color != -color) {
            maybe_ko = false;
        }
    }

    if (!legal) {
        std::cerr << "[c_GoBoard::add_stone] illegal move: pos=" << pos
                  << ", color=" << color << std::endl;
        throw std::exception();
    }

    board[pos] = color;
    for (auto stone : captured_stones) {
        board[stone] = EMPTY;
    }

    // Check if the next move may be a ko
    if (maybe_ko && captured_stones.size() == 1) {
        ko_pos = captured_stones.front();
    } else {
        ko_pos = coord<int>(-INF, -INF);
    }

    consecutive_pass = 0;
    step_cnt += 1;
    last_step_pos = pos;
    last_step_color = color;
    return true;
}

const std::vector <coord<int> > c_GoBoard::get_legal_moves(const int &color){
    // Get all legal moves for a specific color
    // @param color: color of the stones
    // @return: a list of legal moves
    std::vector <coord<int> > legal_moves;

    BreadthFirstSearch bfs(board);
    array2d<char> captureable(n, n);

    auto is_legal = [&](const coord<int> &pos) {
        if (board[pos] != EMPTY || pos == ko_pos) {
            return false;
        }

        for (auto direction : directions) {
            coord<int> neighbor = pos + direction;
            if (!board.is_valid_pos(neighbor)) {
                continue;
            }

            int neighbor_color = board[neighbor];
            if (neighbor_color == EMPTY) {
                // The stone has liberties, this move is legal
                return true;
            }

            if (!bfs.is_visited(neighbor)) {
                // Check if the group has liberties
                auto [stones, captured] = bfs.capture(neighbor, pos);
                if (captured) {
                    for (auto stone : stones) {
                        captureable[stone] = true;
                    }
                }
            }

            if (neighbor_color == color && !captureable[neighbor]) {
                // Our stone has liberties, this move is legal
                return true;
            }
            if (neighbor_color == -color && captureable[neighbor]) {
                // Opponent's stones have no liberties, this move is legal
                return true;
            }
        }

        // No liberties found, this move is illegal
        return false;
    };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            coord<int> pos(i, j);
            if (is_legal(pos)) {
                legal_moves.push_back(pos);
            }
        }
    }

    return legal_moves;
}

coord<float> c_GoBoard::get_score(){
    // Get the score of the game
    // @return: a pair of float, (black score, white score)
    // NOTE: the score is the number of stones and empty points captured by each player
    float scores[2] = {.0, .0};// black, white

    BreadthFirstSearch bfs(board);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            coord<int> pos(i, j);
            int color = board[pos];
            if (color == EMPTY) {
                if (!bfs.is_visited(pos)) {
                    // Determine the belonging of the empty points
                    auto [stones, group_color] = bfs.belong(pos);
                    if (group_color == BLACK) {
                        scores[0] += stones.size();
                    } else if (group_color == WHITE) {
                        scores[1] += stones.size();
                    }
                }
            } else if (color == BLACK) {
                scores[0] += 1;
            } else if (color == WHITE) {
                scores[1] += 1;
            }
        }
    }

    return coord<float>(scores[0], scores[1]); // black, white
}

int c_GoBoard::is_game_over(){
    // Check if the game is over, and return the winner if it is
    // @return: the winner of the game (BLACK, WHITE or DRAW), or EMPTY if the game is not over
    if (step_cnt >= n * n * 2 + 1 || consecutive_pass >= 2){
        auto scores = get_score();
        if (scores.x > scores.y){
            return BLACK;
        }
        else if (scores.x < scores.y){
            return WHITE;
        }
        else{
            return DRAW;
        }
    }
    return EMPTY;
}