=head1 LICENSE
GNU-PL1

=head1 CONTACT

shen-lab

=cut

=head1 NAME

 gMVP

=head1 SYNOPSIS

 mv gMVP.pm ~/.vep/Plugins
 ./vep -i input.vcf --plugin gMVP,/share/mendel/home/yz3419/AnnoDB/gMVP.txt.gz,gMVP,gMVP_rankscore

=head1 DESCRIPTION

 A VEP plugin that retrieves gMVP scores for variants from a tabix-indexed gMVP data file.

 Predicting pathogenicity of missense variants by deep learning

=cut

package gMVP;

use strict;
use warnings;

use Bio::EnsEMBL::Utils::Sequence qw(reverse_comp);

use Bio::EnsEMBL::Variation::Utils::BaseVepTabixPlugin;

use base qw(Bio::EnsEMBL::Variation::Utils::BaseVepTabixPlugin);

my %INCLUDE_SO = map {$_ => 1} qw(missense_variant stop_lost stop_gained start_lost);

sub new {
  my $class = shift;

  my $self = $class->SUPER::new(@_);

  $self->expand_left(0);
  $self->expand_right(0);

  $self->get_user_params();

  return $self;
}

sub feature_types {
  return ['Transcript'];
}

sub get_header_info {
  return { gMVP => 'gMVP score',
           gMVP_rankscore => 'gMVP rankscore'};
}

sub run {
  my ($self, $tva) = @_;

  # only for missense variants
  return {} unless grep {$INCLUDE_SO{$_->SO_term}} @{$tva->get_all_OverlapConsequences};

  my $vf = $tva->variation_feature;

  return {} unless $vf->{start} eq $vf->{end};

  # get allele, reverse comp if needed
  my $allele = $tva->variation_feature_seq;
  reverse_comp(\$allele) if $vf->{strand} < 0;

  return {} unless $allele =~ /^[ACGT]$/;

  # get transcript stable ID
  my $tr_id = $tva->transcript->stable_id;

  my ($res) = grep {
    $_->{pos} == $vf->{start} &&
    $_->{alt} eq $allele &&
    $_->{tr}  eq $tr_id
  } @{$self->get_data($vf->{chr}, $vf->{start}, $vf->{end})};

  return $res ? { gMVP => $res->{gMVP},
                  gMVP_rankscore => $res->{gMVP_rankscore}} : {};
}

sub parse_data {
  my ($self, $line) = @_;

  my @split = split /\t/, $line;

  return {
    pos => $split[1],
    alt => $split[3],
    tr  => $split[6],
    gMVP => $split[13],
    gMVP_rankscore => $split[14]
  };
}

sub get_start {
  return $_[1]->{pos};
}

sub get_end {
  return $_[1]->{pos};
}

1;
