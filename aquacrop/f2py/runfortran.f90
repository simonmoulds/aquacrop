SUBROUTINE F2PY_run(A,B)
    INTEGER :: i, j, k, l
    REAL(8), DIMENSION(100,100,100,100), INTENT(in) :: A
    REAL(8), INTENT(out) :: B

    B = 0.0d0
    DO i=1, 100
        DO j=1, 100
            DO k=1, 100
                DO l=1,100
                   B = B + A(l,k,j,i)
                END DO
            END DO
        END DO
    END DO
END SUBROUTINE F2PY_run
