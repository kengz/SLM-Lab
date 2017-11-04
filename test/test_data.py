import pytest


class TestData:
    '''
    Base class for unit testing data provided to a network
    '''

    def test_zero(self, test_data_gen):
        '''Checks that a batch of data is not zero
        batch: tensor representing a batch of data. May be of arbitrary
        dimension, but first dimension always represents batch size
        returns: true if batch is not all zero, false otherwise'''
        # TODO: implement basic version
        batch = test_data_gen[0]
        assert batch is None

    def test_different(self, test_data_gen):
        '''Checks that batches of data are changing
        returns: true if two batches are not equal to each other,
                 false otherwise'''
        # TODO: decide on format for dataloader and implement basic version
        dataloader = test_data_gen[1]
        assert dataloader is None
