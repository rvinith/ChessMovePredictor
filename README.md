State Values for Chess Pieces
WHITE:
    PAWN : 1
    KNIGHT : 2
    BISHOP : 3
    ROOK : 4
    QUEEN : 5
    KING : 6
BLACK:
    PAWN : 7
    KNIGHT : 8
    BISHOP : 9
    ROOK : 10
    QUEEN : 11
    KING : 12

Point Values (based on Reinfeld Values)
    PAWN : 1
    KNIGHT : 3
    BISHOP : 3
    ROOK : 5
    QUEEN : 9
    KING : 0

Other Conditions:
    Can castle king side (x2 for both colors)
    Can castle queen side  (x2 for both colors)
    Player turn (true for white, false for black -> one bit)
    Can en passant (x2 for both colors)
    Empty space on board (x2 for both colors)

Result:
    1-0 for White win
    0-1 for Black win
    1/2-1/2 for draw

Reward:
    +1 for win
    -1 for loss